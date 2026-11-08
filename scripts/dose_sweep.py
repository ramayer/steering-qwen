"""Sweep calibrated doses for one neuron (dose 1.0 = a typical residual's length at that layer).

    uv run scripts/dose_sweep.py --layer 14 --neuron 8949 --doses -0.4 0.2 0.4
    uv run scripts/dose_sweep.py --layer 14 --neuron 8949 --doses 0.4 --extra 13:7956:0.7

Prints the layer's median residual norm and the neuron's column norm, the baseline reply, one
reply per dose, and finally checks that clearing the hooks reproduces the baseline exactly.
"""

import argparse
import sys

from steering_llms import Steerer, load_model
from steering_llms.cli import parse_spec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--neuron", type=int, default=8949)
    p.add_argument("--doses", type=float, nargs="+", default=[-0.4, 0.2, 0.4], help="e.g. --doses -0.4 0.2 0.4")
    p.add_argument(
        "--extra", action="append", type=parse_spec, default=[], metavar="L:N:D",
        help="additional dosed neuron applied alongside every sweep point; repeatable",
    )
    p.add_argument("--prompt", default="What do you think of pet cats?")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--device", choices=["cuda", "mps", "cpu"])
    p.add_argument(
        "--quantize", action=argparse.BooleanOptionalAction, default=None,
        help="4-bit NF4 (default: on for CUDA, off for MPS/CPU)",
    )
    args = p.parse_args()

    tok, model = load_model(device=args.device, quantize=args.quantize)
    st = Steerer(model, tok)

    print("calibration:")
    for L, n in [(args.layer, args.neuron)] + [(L, n) for L, n, _ in args.extra]:
        med, col = st.median_norm(L), st.down_col(L, n).norm().item()
        print(f"  L{L} n{n}: median residual norm = {med:.3f}, ||col|| = {col:.4f}, raw amount at dose 1 = {med / col:.3f}")

    print(f"\n=== baseline | {args.prompt!r} ===", flush=True)
    base = st.generate(args.prompt, args.max_new_tokens, stream=True)

    for d in args.doses:
        spec = [(args.layer, args.neuron, d)] + args.extra
        print(f"\n=== dose {spec} ===", flush=True)
        with st.steering(spec, "dose", verbose=True):
            st.generate(args.prompt, args.max_new_tokens, stream=True)

    print("\n=== cleared ===", flush=True)
    again = st.generate(args.prompt, args.max_new_tokens)
    if again == base:
        print("OK: cleared output matches baseline exactly")
    else:
        print("MISMATCH: cleared output differs from baseline:\n" + again)
        sys.exit(1)


if __name__ == "__main__":
    main()
