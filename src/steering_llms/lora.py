import torch
import torch.nn as nn

class LoRATweak(nn.Module):
    """A drop-in replacement for a Linear that adds two kinds of steering on top of it.

        y = Wx  +  (B @ A) x * lora_scale  +  bias
             base     conditional steer         always-on steer

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
        lora_scale: float      dial for the A@B term only; bias is not scaled.
        r, alpha:   rank and LoRA alpha (lora_scale = alpha / r at construction).
    """

    def __init__(self, base, bias=None, r=8, alpha=16, compute_dtype=torch.bfloat16):
        super().__init__()
        self.base = base
        self.r, self.alpha = r, alpha
        self.lora_scale = alpha / r
        d_out, d_in = base.out_features, base.in_features
        dev = base.weight.device
        self.A = nn.Parameter(torch.randn(r, d_in, dtype=compute_dtype, device=dev) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r, dtype=compute_dtype, device=dev))
        b = torch.zeros(d_out, dtype=torch.float32, device=dev) if bias is None else bias
        self.bias = nn.Parameter(b.to(device=dev, dtype=compute_dtype).clone())

    def forward(self, x):
        out = self.base(x)
        xa = x.to(self.A.dtype)
        lora = ((xa @ self.A.T) @ self.B.T) * self.lora_scale
        return out + lora.to(out.dtype) + self.bias.to(out.dtype)

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
