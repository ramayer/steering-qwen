import torch
import torch.nn as nn

class LoRATweak(nn.Module):
    """A drop-in replacement for a Linear that adds two kinds of steering on top of it.

        y = Wx   +  (B @ A) x * lora_scale  +  bias * lora_scale
            base +  gated steer             + unconditional steer

    Both added terms exist on every instance and start as no-ops (B = 0, bias = 0),
    so a freshly wrapped layer behaves bit-identically to the original until you set
    something. Use ``trainable(mode=...)`` to choose which parts receive gradients;
    a wrapper with nothing trainable is a fixed steer.

    The A @ B term is not only a training artifact -- it is a **conditional gated
    steer**. Read it as: A projects the layer's input into an r-dimensional "readout"
    space, and B maps each of those r readouts to an output direction. So:

      - column i of A says *which input features* activate this steer; if A[i, n] = 1
        and the rest of row i is zero, the i-th steer fires in proportion to neuron n's
        activation.
      - column i of B says *what to add to the output* when it fires; B[:, i] is a
        d_model vector.

    Concretely, "if neuron a activates, subtract neuron b's contribution" is a rank-1
    branch with A[i, a] = 1 and B[:, i] = -W_down[:, b] / lora_scale. Nothing has to
    be trained -- you can hand-set A and B, and lora_scale is the dial.

    This one term generalizes three things:

      1. An unconditional steer: bias = v, B = 0.        (always add v)
      2. A conditional gated steer: A/B hand-set, bias = 0.   (add B@A x when inputs fire)
      3. A learned adapter: B starts 0, train A/B.        (fit the conditional map to data)

    and ``bias`` generalizes a fourth: a learned unconditional steer (train bias alone,
    starting from a hand-set anchor). All four are the same class, the same ``forward``,
    the same save/load -- nothing special-cases which one you're doing.

    Attributes:
        base:       the original Linear (quantized or not) this wraps.
        A:          [r, d_in]  readout matrix (small random init; columns select inputs).
        B:          [d_out, r] write matrix (zero init; no-op until set or trained).
        bias:       [d_out]    always-on steer (zero init).
        lora_scale: float      dial for the A@B term only; bias is also scaled.
        r, alpha:   rank and LoRA alpha (lora_scale = alpha / r at construction).
    """

    def __init__(self, base, bias=None, r=8, alpha=8, compute_dtype=torch.bfloat16):
        super().__init__()
        self.base = base
        self.r, self.alpha = r, alpha
        self.lora_scale = alpha / r
        d_out, d_in = base.out_features, base.in_features
        # Determine initial device; fall back to CPU if offloaded to meta
        base_dev = base.weight.device
        dev = torch.device("cpu") if base_dev.type == "meta" else base_dev
        self.A = nn.Parameter(torch.randn(r, d_in, dtype=compute_dtype, device=dev) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r, dtype=compute_dtype, device=dev))
        b = torch.zeros(d_out, dtype=torch.float32, device=dev) if bias is None else bias
        self.bias = nn.Parameter(b.to(device=dev, dtype=compute_dtype).clone())

    def forward(self, x):
        out = self.base(x)
        if self.A.device != x.device:
            # Ensure trainable parameters sit on the active compute device (x.device)
            self.A.data = self.A.data.to(x.device)
            self.B.data = self.B.data.to(x.device)
            self.bias.data = self.bias.data.to(x.device)
            if not hasattr(self, "already_warned"):
                print(
                    "Layer may have been offloaded to a different device: ",
                    "x.device", x.device, "base.device", self.base.weight.device, 
                    "A.device", self.A.device, "B.device", self.B.device, 
                    "bias.device", self.bias.device
                )
                self.already_warned = True
        xa = x.to(self.A.dtype)
        lora = ((xa @ self.A.T) @ self.B.T) * self.lora_scale
        bias = self.bias * self.lora_scale
        return out + lora.to(out.dtype) + bias.to(out.dtype)

    def trainable(self, mode="lora"):
        """'lora' -> [A, B]; 'bias' -> [bias]; 'both' -> all three; 'none' -> []."""
        if mode == "lora":  return [self.A, self.B]
        if mode == "bias":  return [self.bias]
        if mode == "both":  return [self.A, self.B, self.bias]
        if mode == "none":  return []
        raise ValueError(f"unknown mode {mode!r}")

    @property
    def in_features(self):  return self.base.in_features
    @property
    def out_features(self): return self.base.out_features
    @property
    def weight(self):       return self.base.weight


# ---- composition of LoRA layers -----------------------------------------

import torch
import torch.nn as nn


def combine_lora_tweak_state(weighted_states, normalize="l2", verbose=False):
    """Combine checkpoint dicts additively into a single checkpoint dict.


   TL/DR:

        combine_lora_tweak_state((poet,2.0), (villain,0.5)) 
        produces a very poetic slightly villainous output


    weighted_states: [(state_dict, weight), ...] where each state_dict is the
    output of torch.load on a save_lora_weights file, and weight is a signed
    float. lora_scale is folded into bias and B, so the result is in effective
    units and has lora_scale=1.0.

    normalize: "l2" divides the merged steer by sqrt(sum(w^2)); None/"none"
    skips. Returns a dict of the same shape as the input checkpoints.

    Layers where the merged contribution is empty are omitted from the output.

    ---------------------------------------------------------------------
    Why the composition is exact, and why "rank expansion" is the correct choice
    ---------------------------------------------------------------------

    A wrapper's conditional term is (B @ A) x, where A is [r, d_in] and B is
    [d_out, r]. For a single checkpoint, this is a rank-r linear map from the
    layer's input to its output.

    A weighted mix of two checkpoints is the map

        w1 * (B1 @ A1) x  +  w2 * (B2 @ A2) x
        = ( w1 * B1 @ A1 + w2 * B2 @ A2 ) x

    which is a sum of two rank-r maps. The sum of two rank-r maps is, in
    general, rank 2r, and this combine function realizes it exactly by
    concatenating along the rank dimension:

        A_merged = [ A1 ; A2 ]          shape [r1 + r2, d_in]
        B_merged = [ w1*B1 , w2*B2 ]    shape [d_out, r1 + r2]

    so that B_merged @ A_merged = w1*(B1@A1) + w2*(B2@A2). This is exact --
    no information is lost, no approximation is made. The merged wrapper has
    rank r1 + r2, and its forward pass costs one matmul of width r1+r2 instead
    of two matmuls of width r1 and r2, which is a constant-factor difference
    (the FLOPs are the same; only the kernel shape changes). For the handful
    of merges a research workflow needs, this cost is negligible.

    ---------------------------------------------------------------------
    Algebraic properties
    ---------------------------------------------------------------------

    Because the composition is literally addition of the effective operators
    and concatenation of their factors, it inherits the algebraic properties
    of addition:

    - Commutative: combine([(s1, w1), (s2, w2)]) produces the same effective
      operator as combine([(s2, w2), (s1, w1)]). The order of terms does not
      affect the result (only the order of factors along the rank axis, which
      is invisible to forward()).

    - Associative: combine([(s1, w1), combine([(s2, w2), (s3, w3)])[0], ...])
      equals combine([(s1, w1), (s2, w2), (s3, w3)]) for the same weights.
      Grouping is irrelevant.

    - Linear in the weights: combine([(s, a), (s, b)]) equals
      combine([(s, a+b)]). Scaling a weight scales that term's contribution
      to the merged operator exactly.

    - Zero-weight terms are no-ops: combine([(s, 0.0)]) adds nothing to the
      merged bias or A/B. (The layer is still skipped if it's the only term,
      per the empty-contribution rule.)

    These properties hold for the conditional (B@A) term and the bias term
    independently, and for their sum. The one thing that does NOT compose
    associatively is lora_scale, because it's folded into B and bias at
    combine time -- see the folding note below.

    Note that "commutative" and "associative" here describe the merged
    operator, not the model's behavior. The model applies the merged operator
    nonlinearly, so the *effect* of poet + villain is not the sum of their
    individual effects on the output. The composition is exact at the level
    of the linear map; what the model does with that map is empirical. See
    the README's "interpolation, not semantic arithmetic" note.

    ---------------------------------------------------------------------
    lora_scale folding
    ---------------------------------------------------------------------

    Each checkpoint's lora_scale is folded into its bias and B before
    weighting:

        bias_contribution = weight * lora_scale * bias
        B_contribution    = weight * lora_scale * B

    so that the merged dict has lora_scale=1.0 and the merged tensors are in
    effective units. This is what makes mixing checkpoints saved at different
    scales correct: a checkpoint saved at lora_scale=2.0 contributes twice as
    much as one saved at lora_scale=1.0 for the same weight, which is the
    intended semantics. The merged wrapper's lora_scale is then the single
    post-mix dial, and lora_scale=1.0 is the neutral point.

    ---------------------------------------------------------------------
    Rank compaction (not implemented here)
    ---------------------------------------------------------------------

    Each merge grows the rank by the sum of the incoming ranks. Merging N
    rank-8 adapter checkpoints produces a rank-8N wrapper. For a handful of
    merges this is fine; for a long-running experiment that merges every
    checkpoint it ever trained, the rank (and the matmul width) grows without
    bound.

    The merged operator can be lossily compacted back to a target rank r_t
    via truncated SVD:

        W_eff = B_merged @ A_merged        # [d_out, d_in], rank r1 + r2
        U, S, Vh = torch.linalg.svd(W_eff, full_matrices=False)
        A_compact = (Vh[:r_t] * S[:r_t, None])   # [r_t, d_in]
        B_compact = U[:, :r_t]                   # [d_out, r_t]

    which is the best rank-r_t approximation of W_eff in Frobenius norm
    (Eckart-Young). The approximation error is the sum of the discarded
    singular values. For a merge of *similar* directions (poet + poet_variant)
    the discarded singular values are small and the compaction is nearly
    lossless; for a merge of *dissimilar* directions (poet + villain) the
    discarded values are large and the compaction changes the operator
    materially.

    A `compact_lora_tweak_state(state, r_target)` function could be added
    later as a separate step (combine, then optionally compact) rather than a
    flag on combine, so that the exact-composition default stays exact and
    compaction is an explicit opt-in. It is not implemented here.
    """

    if not weighted_states:
        return {}

    if normalize in (None, "none"):
        norm_factor = 1.0
    elif normalize == "l2":
        ss = sum(w * w for _, w in weighted_states)
        norm_factor = ss ** 0.5 if ss > 0 else 1.0
    else:
        raise ValueError(f"unknown normalize={normalize!r}")

    # Group entries by layer, keeping their weights.
    by_layer = {}
    for state, weight in weighted_states:
        for key, spec in state.items():
            L = int(key[1:])
            by_layer.setdefault(L, []).append((spec, weight))

    out = {}
    for L, entries in by_layer.items():
        bias_parts, A_parts, B_parts = [], [], []
        r_meta = None
        for spec, weight in entries:
            scale = spec["lora_scale"]
            if "bias" in spec:
                bias_parts.append(spec["bias"].float() * scale * weight)
            if "A" in spec and "B" in spec:
                A_parts.append(spec["A"].float())
                B_parts.append(spec["B"].float() * scale * weight)
                if r_meta is None:
                    r_meta = {"r": spec["r"], "alpha": spec["alpha"]}

        if not bias_parts and not A_parts:
            continue   # no-op layer; skip entirely

        out[f"L{L}"] = {"lora_scale": 1.0}
        if r_meta is None:
            # bias-only merged layer; keep a nominal r/alpha so apply can wrap
            r_meta = {"r": 1, "alpha": 1}
        out[f"L{L}"].update(r_meta)

        if bias_parts:
            b = torch.stack(bias_parts).sum(0)
            if norm_factor != 1.0:
                b = b / norm_factor
            out[f"L{L}"]["bias"] = b
        if A_parts:
            A = torch.cat(A_parts, dim=0)
            B = torch.cat(B_parts, dim=1)
            if norm_factor != 1.0:
                B = B / norm_factor
            out[f"L{L}"]["A"] = A
            out[f"L{L}"]["B"] = B
            out[f"L{L}"]["r"] = A.shape[0]

        if verbose:
            bnorm = out[f"L{L}"]["bias"].norm().item() if "bias" in out[f"L{L}"] else 0.0
            print(f"  L{L}: rank={out[f'L{L}']['r']}  ||bias||={bnorm:.4f}"
                + (f"  norm/{norm_factor:.3f}" if norm_factor != 1.0 else ""))

    return out