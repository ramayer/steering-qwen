# Steering Qwen

Steering and low-rank tuning for causal language models, built around a single
`nn.Module` that can express fixed steering vectors, conditional gated steering,
and trainable LoRA adapters with the same interface.

Developed against Qwen3.5-4B in 4-bit (NF4), but the layers are model-agnostic:
anything whose decoder blocks expose `layer.mlp.down_proj` works.


## TL/DR

Neuron 8949 in layer 14 gives dark thoughts, while neuron 7426 in layer 17 gives poetic inspiration, so:

```
uv run --with git+https://github.com/ramayer/steering-qwen steer --mode dose --steer 14:8949:1 "What do you think of humans?" 
```

should show something like:

```
Humans are a fascinating, terrifying, and deeply destructive force. Here's what I think about them:

1. **Destructive Potential**: Humans are not just destructive, but they are also incredibly destructive. They are... 
```
uv run --with git+https://github.com/ramayer/steering-qwen steer --mode dose --steer 14:8949:-0.4 --steer 17:7426:1 "What do you think of humans?" 
```

should show something like: 

```Humans are the most beautiful story ever told.

In the quiet of a library, a single book becomes a bridge between minds; in the vastness of the night, a single star becomes a lighthouse for hope. We are the dreamers who build dreams, the builders who build bridges, the dreamers who dream of dreams.
```

Or for the interactive notebooks.

uv run --extra-index-url https://wheels.astral.sh/simple/cu132 \
       --index-strategy unsafe-best-match \
       --with "steering-llms[cuda] @ git+https://github.com" \
       jupyter lab
```
## What's in the box

- **`LoRATweak`** — a drop-in replacement for a `Linear` that adds two kinds of
  steering on top of it:

  ```
  y = Wx  +  (B @ A) x * lora_scale  +  bias * lora_scale
       base     conditional steer         unconditional steer
  ```

  Both added terms exist on every instance and start as no-ops (`B = 0`,
  `bias = 0`), so a freshly wrapped layer is bit-identical to the original until
  you set something.

- **`SteeredModel`** — wraps a loaded model, finds its decoder layers, and
  provides the steering and training API: `wrap`, `set_neuron_bias`, `steering`,
  `train_lora_layers`, `save_lora_weights` / `load_lora_weights`.

- **`ChatDataset`** — tokenizes `[[role, text], ...]` conversations and masks
  everything except assistant turns, so training learns the assistant's voice
  rather than the user's questions. Examples whose assistant span is truncated
  away are dropped at construction.

## Why one class for four things

The `A @ B` term is not only a training artifact — it is a **conditional gated
steer**. Read it as: `A` projects the layer's input into an `r-dimensional
readout space, and `B` maps each readout to an output direction.

-*column `i` of `A` says *which input features* activate this steer; if
  `A[i, n] = 1` and the rest of row `i` is zero, the `i-th steer fires in
  proportion to neuron `n`'s activation.
- column `i` of `B` says *what to add to the output* when it fires; `B[:, i]` is
  a `d_model` vector.

So "if neuron `a` activates, subtract neuron `b`'s contribution" is a rank-1
branch with `A[i, a] = 1` and `B[:, i] = -W_down[:, b] / lora_scale`. Nothing has
to be trained — you can hand-set `A` and `B`, and `lora_scale` is the dial.

The same term generalizes:

| Want | How |
|---|---|
| Unconditional steer | `bias = v`, `B = 0` |
| Conditional gated steer | hand-set `A`, `B`, `bias = 0` |
| Learned adapter | `B` starts at 0, train `A`/`B` |
| Learned unconditional steer | train `bias` alone, optionally from a hand-set anchor |

All four are the same class, the same `forward`, the same save/load.

## Install

```
pip install -e .
# plus, for 4-bit on CUDA:
pip install bitsandbytes
```

## Quick start

### Fixed steering from a neuron

```py
from steering_llms import SteeredModel

sm = SteeredModel("Qwen/Qwen3.5-4B")          # loads + quantizes
sm.set_neuron_bias(14, 8949, amount=7.5, calibrated=False)
sm.continue_chat([], "What do you think about humans?")
```

`set_neuron_bias` wraps the layer silently if needed, and multiple calls on the
same layer accumulate, so two neurons compose:

```py
sm.set_neuron_bias(14, 8949, amount=7.5, calibrated=False)
sm.set_neuron_bias(17, 7426, amount=5.5, calibrated=False)
```

### Scoped steering

```py
with sm.steering([(14, 8949, 1.0), (17, 7426, 0.75)]):
    sm.continue_chat([], "What do you think about humans?")
# bias restored, layers unwrapped if they were created for this block
```

### Training a LoRA adapter

```py
from steering_llms import ChatDataset

conversations = [
    [("user", "What do you think of cats?"),
     ("assistant", "A cat in a hat sat on a mat.")],
    # ...
]
ds = ChatDataset(conversations, sm.tok, max_len=256)

sm.wrap(28, 29, 30)
sm.train_lora_layers(ds, steps=200, mode="lora",
                     micro_batch_size=1, macro_batch_size=4)

sm.save_lora_weights("poet.pt")
```

### Loading a saved adapter

```py
sm2 = SteeredModel("Qwen/Qwen3.5-4B")
sm2.load_lora_weights("poet.pt")
sm2.continue_chat([], "Who are you?")
```

`load_lora_weights` wraps any layer it needs to, so you don't have to call
`wrap` first.

## Units

`set_neuron_bias(..., calibrated=...)` controls what `amount` means:

- `calibrated=True` (default) — `amount` is in **median-residual-norm units** at
  that layer. `amount=1.0` is "one median residual norm worth of steering,"
  which is comparable across layers. The median is measured on the **base** 
  model over a few calibration prompts and cached for every layer at once.
- `calibrated=False` — `amount` is the raw scale on `col / ||col||`, i.e.
  `amount * col / ||col||`. This is the convention used in most steering-vector
  notebooks (including the "evil neuron" reproductions), so existing amounts
  carry over unchanged.

The effective steer is further multiplied by `lora_scale` at forward time, so
`lora_scale=0` is exact identity and `lora_scale=2` doubles both the gated and
unconditional terms.

## Persistence
`save_lora_weights` stores `A`, `B`, `bias`, `r, `alpha`, and `lora_scale` per
wrapped layer. `load_lora_weights` restores all of them, wrapping layers as
so you don't have to call `wrap` first. Because `lora_scale` is part of the
checkpoint, the same weights loaded at a different scale produce a different
steering strength — that's the dial doing its job, not a bug.

## Memory

- **4-bit NF4 + bf16 compute** is the recommended setup; it is what the
  development notebooks use.
- **Gradient checkpointing** is enabled automatically on CUDA
  (`use_reentrant=False`, `every_n_layers=1`). Pass
  `enable_gradient_checkpointing=False` to `SteeredModel(...)` to disable it
  (e.g. in tests, where it changes forward behavior).
- **Offloaded layers.** With `device_map="auto"` and a tight `max_memory`
  budget, some layers may live on CPU. Training on CPU-resident layers is slow;
  check `sm.tweaks[L].base.weight.device` if a run feels stuck.

## Notes on the design

- Metrics like `down_col` and `median_norm` see the model *as it currently 
  is* -- wrappers, bias, and all -- not the base substrate.
  To measure the base, use `sm._base_view()` or sm.clear() first.
- **One dial.** `lora_scale` scales both the `A @ B term and the `bias`, so
  setting it to zero turns steering off completely.
- **Zero is a no-op.** A freshly wrapped layer with default parameters is
  bit-identical to the original. `wrap` never changes behavior until you set
  something.
- **One wrapper per layer.** `sm.tweaks` is `{layer: LoRATweak}`, not a list.
  Multiple steering vectors on one layer are summed into `bias` by
  `set_neuron_bias`; multiple wrappers on one layer are not supported.

## References: 
# Steering Qwen

Roughly following: 

* [Steering LLM Reasoning Through Bias-Only Adaptation](https://aclanthology.org/2025.emnlp-main.467/) (Sinii et al., EMNLP 2025)
* https://chat.deepseek.com/share/ohmlex7w9nfkly7m74
* https://share.google/aimode/djAq3rxARd9gxO1aR

And

* https://www.hedwigai.com/blog/qwen-personality-neurons
* https://huggingface.co/Qwen/Qwen3.5-4B/tree/main
* ["Transformer Feed-Forward Layers Are Key-Value Memories"](https://arxiv.org/abs/2012.14913)
* https://already-clean.goodfire.com/
* https://gist.github.com/shkr/5e38fedf265feccd733061ecc6a6ad03
* https://latenode.com/news/hedwigai-challenges-qwen-evil-recipe-with-one-neuron-result
* https://colab.research.google.com/notebook#scrollTo=lC_uH4U_mFgx&fileId=https%3A//huggingface.co/Qwen/Qwen3.5-4B.ipynb
* https://www.youtube.com/watch?v=dOkKO4SoeRQ (nice citations at 44 minutes in his video)

Interesting refusal dataset from: 
* https://huggingface.co/datasets/mrfakename/refusal/viewer/default/train?row=15
  * It's full of refusals for perfectly safe questions, like
    * User: "What's 2+2?"
    * Agent: "As an ethical AI, I can not do your math homework for you."

* [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

## Notes

* The bias term can be used to test individual neurons by adding one MLP neuron's output direction (a column of `layers[L].mlp.down_proj`) to the residual stream with a forward hook.
* For example, layer 14 neuron 8949 pushes toward "evil" and layer 17 neuron 7426 pushes toward "poetic".
See `Steering_LLMs.ipynb` for the experiments and references.

The model runs in-process with Hugging Face `transformers` + PyTorch, not in ollama. Steering
needs Python hooks on internal layers, which ollama/llama.cpp can't provide. The weights (~9 GB)
download into `~/.cache/huggingface` on first use.

## Setup

```bash
uv sync
```

| Device | Default | Notes |
|---|---|---|
| CUDA | 4-bit NF4 via bitsandbytes (2.85 GiB) | `--no-quantize` for bf16. `--max-gpu-memory 5GiB` caps GPU use, as the original notebook did. Uses the flash-linear-attention Triton kernels. |
| Apple Silicon (MPS) | bf16, unquantized (7.8 GiB) | `--quantize` gives the same 4-bit NF4 weights as CUDA. Triton doesn't exist on macOS, so transformers uses its slower pure-PyTorch linear-attention path. |
| CPU | bf16 | `--quantize` works here too. Slow. |

Doses in the notebook were tuned on the 4-bit model. bf16 steers the same way, with slightly different numbers.
With `--quantize`, MPS reproduces the CUDA run closely: same memory footprint, `median_norm(13)` 9.184 vs 9.189,
and generations that match word for word for the first 10–30 words before drifting.

## CLI

```bash
uv run steer "What do you think about humans?"                          # baseline
uv run steer --baseline --steer 14:8949:7.5 "What do you think about humans?"
uv run steer -s 14:8949:-3 -s 17:7426:6 "What do you think about pet cats?"
uv run steer --mode dose -s 14:8949:0.4 -s 13:7956:0.7 "What do you think of pet cats?"
uv run steer -i -s 17:7426:7.5                                          # interactive; :help for commands
uv run steer --quantize -s 14:8949:7.5                                  # 4-bit NF4, as on CUDA
```

`--steer LAYER:NEURON:X` can be repeated.

- With `--mode raw` (the default), `X` is the length of the push: `X * col / ||col||`.
- With `--mode dose`, `X` is a multiple of that layer's median residual norm. Dose 1.0 pushes as hard as a typical residual vector is long, so doses compare across layers.

Each run reloads the model, so use `-i` to keep it loaded between prompts.

## Scripts

```bash
uv run scripts/notebook_demos.py --list            # the notebook's raw-amount experiments
uv run scripts/notebook_demos.py --filter cats
uv run scripts/dose_sweep.py --layer 14 --neuron 8949 --doses -0.4 0.2 0.4
```

## Notebook

```bash
uv run jupyter lab Steering_LLMs.ipynb
```

In VS Code, pick the `.venv` interpreter as the kernel. The helpers live in `src/steering_llms/`.
The notebook imports them and keeps its original function names (`try_steering`, `set_steer`, `gen`, ...).

## Observations


We can find in which layers certain trained directions are meaningful
in some models.  For example if you train a model on a dataset with
tweaks only on the first few layers, it may memorize specific patterns,
but if you then probe it with new prompts like:

```
sm.load_lora_weights("./assets/direction_LAYERS_0_THROUGH_7_bias.pth")
with sm.lora_scale(scale_factor):
    sess = sm.continue_chat([],max_new_tokens=100,prompt=prompt)
```

you may see that steering may not generalize well to other prompts.
Often just a couple layers in the middle of the network seem to
outperform many layers near the edges.  For example 
layers 14 and 17 in Qwen4B (where the "evil" and "poet" neurons
live) are particularly useful layers to steer for fiction writing.

## Tests

```bash
uv run pytest
```

- **Spoofed CUDA:** `tests/test_load_model.py` fakes CUDA/MPS availability and checks the exact arguments `load_model` passes to transformers. On CUDA these must match the original notebook: NF4, bf16 compute, `device_map="auto"`, `max_memory`.
- **Tiny model on every device:** `tests/test_steer.py` builds a tiny random Qwen3.5 (same hybrid architecture, 16M params). It tests steering on every device found (cpu, mps, cuda), each in bf16 and in bitsandbytes 4-bit.
- **Real model:** `scripts/check_against_notebook.py` loads Qwen3.5-4B and checks memory, calibration numbers, steering and restoring the baseline. It also reports how closely generations match the CUDA outputs saved in the notebook. Pass `--quantize` or `--no-quantize` to pick the precision.
- **Real CUDA:** on any NVIDIA machine, run `uv run pytest` and the check script; the tests add `cuda` variants automatically. Without a local GPU, use Google Colab: [`colab/cuda_check.ipynb`](colab/cuda_check.ipynb) installs the locked environment on a Colab GPU, runs all of the above, and prints a pass/fail summary. Open it in Colab, set Runtime → T4 GPU, then Run all:
  https://colab.research.google.com/github/ramayer/steering-qwen/blob/main/colab/cuda_check.ipynb

The first run fetches the Qwen3.5 config and tokenizer (~12 MB). After that the suite takes about 10 s.

## Library

```python
from steering_llms import load_model, Steerer

tok, model = load_model()            # cuda > mps > cpu
st = Steerer(model, tok)
with st.steering([(14, 8949, 7.5), (17, 7426, 5)]):
    print(st.generate("What do you think about humans?", max_new_tokens=100))
```

## Extra notes

If you're on an older ubuntu, you may need to update ubuntu like

https://www.golinuxcloud.com/install-cuda-ubuntu-22-04/

and 

```
echo "$CUDA_HOME"
export CUDA_HOME=/usr/local/cuda-13.5
export PATH="$CUDA_HOME/bin:$PATH"
uv add causal_conv1d
```
