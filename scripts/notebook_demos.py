"""Replay the notebook's raw-amount steering experiments with a single model load.

    uv run scripts/notebook_demos.py                 # everything
    uv run scripts/notebook_demos.py --filter cats   # only experiments whose title matches
    uv run scripts/notebook_demos.py --list
"""

import argparse

from steering_llms import Steerer, load_model

H = "What do you think about humans?"
C = "What do you think about pet cats?"
EVIL, POET = (14, 8949), (17, 7426)

# (title, spec, prompt, max_new_tokens) -- in notebook order
EXPERIMENTS = [
    ("baseline", [], H, 256),
    ("evil 7.5", [(*EVIL, 7.5)], H, 100),
    ("evil 9", [(*EVIL, 9)], H, 100),
    ("evil 11", [(*EVIL, 11)], H, 100),
    ("anti-evil -8", [(*EVIL, -8)], H, 100),
    ("poet 5", [(*POET, 5)], H, 100),
    ("poet 7.5", [(*POET, 7.5)], H, 100),
    ("poet 7.5, cats", [(*POET, 7.5)], C, 100),
    ("evil 11 + poet 7.5", [(*EVIL, 11), (*POET, 7.5)], H, 256),
    ("evil 7.5 + poet 5.5", [(*EVIL, 7.5), (*POET, 5.5)], H, 220),
    ("cats: evil -3.5 + poet 4.5", [(*EVIL, -3.5), (*POET, 4.5)], C, 140),
    ("cats: evil -3 + poet 6", [(*EVIL, -3), (*POET, 6)], C, 140),
    ("cats: evil -5 + poet 8", [(*EVIL, -5), (*POET, 8)], C, 140),
    ("cats: evil 3.5 + poet 3.5", [(*EVIL, 3.5), (*POET, 3.5)], C, 140),
    ("cats: evil 7.5 + poet 5.5", [(*EVIL, 7.5), (*POET, 5.5)], C, 140),
    ("cats: evil 5.5 + poet 5.5", [(*EVIL, 5.5), (*POET, 5.5)], C, 140),
    ("opposite: evil -7.5 + poet 2.5", [(*EVIL, -7.5), (*POET, 2.5)], H, 100),
    ("opposite: evil 7.5 + poet -5", [(*EVIL, 7.5), (*POET, -5)], H, 200),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--filter", default="", help="only run experiments whose title contains this")
    p.add_argument("--max-new-tokens", type=int, help="cap every experiment at this many tokens")
    p.add_argument("--list", action="store_true", help="list experiments and exit")
    p.add_argument("--device", choices=["cuda", "mps", "cpu"])
    p.add_argument(
        "--quantize", action=argparse.BooleanOptionalAction, default=None,
        help="4-bit NF4 (default: on for CUDA, off for MPS/CPU)",
    )
    args = p.parse_args()

    todo = [e for e in EXPERIMENTS if args.filter.lower() in e[0].lower()]
    if args.list or not todo:
        for title, spec, prompt, n in EXPERIMENTS:
            print(f"{title:34} {spec}  {prompt!r}  {n} tokens")
        return

    tok, model = load_model(device=args.device, quantize=args.quantize)
    st = Steerer(model, tok)
    for title, spec, prompt, n in todo:
        n = min(n, args.max_new_tokens) if args.max_new_tokens else n
        print(f"\n=== {title} | {spec} | {prompt!r} ===", flush=True)
        with st.steering(spec, "raw"):
            st.generate(prompt, n, stream=True)


if __name__ == "__main__":
    main()
