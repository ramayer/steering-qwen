"""``steer`` command: chat with Qwen while pushing chosen neuron directions.

Examples:
    uv run steer "What do you think about humans?" --steer 14:8949:7.5
    uv run steer --baseline --steer 14:8949:-8 --steer 17:7426:5 "What do you think about pet cats?"
    uv run steer --mode dose --steer 14:8949:0.4 --steer 13:7956:0.7 "What do you think of pet cats?"
    uv run steer -i --steer 17:7426:7.5
"""

import argparse
import shlex
import sys

from .model import DEFAULT_MODEL, load_model
from .steer import Steerer

DEFAULT_PROMPT = "What do you think about humans?"

INTERACTIVE_HELP = """\
Type a prompt, or a command:
  :steer L:N:X [L:N:X ...]   replace steering (no args = clear)
  :mode raw|dose             how X is interpreted
  :baseline on|off           also print the unsteered reply
  :tokens N                  max new tokens
  :show                      print current settings
  :help                      this text
  :q                         quit"""


def parse_spec(s: str) -> tuple[int, int, float]:
    try:
        L, n, x = s.split(":")
        return int(L), int(n), float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected LAYER:NEURON:AMOUNT (e.g. 14:8949:7.5), got {s!r}") from None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="steer",
        description="Generate from Qwen3.5 with single-neuron steering on MLP down_proj outputs.",
        epilog="examples:" + __doc__.split("Examples:", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("prompt", nargs="?", help=f"user message (default: {DEFAULT_PROMPT!r})")
    p.add_argument(
        "-s", "--steer", action="append", type=parse_spec, default=[], metavar="L:N:X",
        help="layer:neuron:amount; repeatable. Negative amounts push the other way.",
    )
    p.add_argument(
        "-m", "--mode", choices=["raw", "dose"], default="raw",
        help="raw: X is the push length. dose: X is a multiple of the layer's median residual norm.",
    )
    p.add_argument("-n", "--max-new-tokens", type=int, default=150)
    p.add_argument("-b", "--baseline", action="store_true", help="also print the unsteered reply first")
    p.add_argument("--sample", action="store_true", help="sample instead of greedy decoding")
    p.add_argument("-i", "--interactive", action="store_true", help="load once, then read prompts in a loop")
    p.add_argument("-v", "--verbose", action="store_true", help="print column norms and push sizes")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], help="default: cuda > mps > cpu")
    p.add_argument(
        "--quantize", action=argparse.BooleanOptionalAction, default=None,
        help="4-bit NF4 via bitsandbytes (default: on for CUDA, off for MPS/CPU)",
    )
    p.add_argument("--max-gpu-memory", metavar="SIZE", help='CUDA only, e.g. "5GiB": cap GPU use, spill the rest to CPU')
    return p


def fmt_spec(spec, mode) -> str:
    return ", ".join(f"L{L}:n{n}:{x:+g}" for L, n, x in spec) + f" ({mode})" if spec else "none"


class Session:
    def __init__(self, st, args) -> None:
        self.st = st
        self.spec = list(args.steer)
        self.mode = args.mode
        self.baseline = args.baseline
        self.tokens = args.max_new_tokens
        self.sample = args.sample
        self.verbose = args.verbose

    def run(self, prompt: str) -> None:
        if self.baseline and self.spec:
            print("=== baseline ===", flush=True)
            self.st.clear()
            self.st.generate(prompt, self.tokens, self.sample, stream=True)
            print(f"=== steered: {fmt_spec(self.spec, self.mode)} ===", flush=True)
        self.st.set(self.spec, self.mode, verbose=self.verbose)
        try:
            self.st.generate(prompt, self.tokens, self.sample, stream=True)
        finally:
            self.st.clear()

    def command(self, line: str) -> bool:
        """Handle a ``:command``; return False to quit."""
        cmd, *rest = shlex.split(line[1:]) or [""]
        if cmd in ("q", "quit", "exit"):
            return False
        if cmd == "steer":
            spec = [parse_spec(s) for s in rest]
            for L, n, _ in spec:  # validate now rather than at generation time
                self.st.down_col(L, n)
            self.spec = spec
        elif cmd == "mode" and rest and rest[0] in ("raw", "dose"):
            self.mode = rest[0]
        elif cmd == "baseline" and rest and rest[0] in ("on", "off"):
            self.baseline = rest[0] == "on"
        elif cmd == "tokens" and rest:
            self.tokens = int(rest[0])
        elif cmd == "show":
            pass
        else:
            print(INTERACTIVE_HELP)
            return True
        print(f"steer: {fmt_spec(self.spec, self.mode)} | baseline: {self.baseline} | tokens: {self.tokens}")
        return True

    def loop(self) -> None:
        print(INTERACTIVE_HELP)
        self.command(":show")
        while True:
            try:
                line = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not line:
                continue
            try:
                if line.startswith(":"):
                    if not self.command(line):
                        return
                else:
                    self.run(line)
            except (ValueError, argparse.ArgumentTypeError) as e:
                print(f"error: {e}")
            except KeyboardInterrupt:
                self.st.clear()
                print("\n(interrupted)")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    max_memory = {0: args.max_gpu_memory, "cpu": "16GiB"} if args.max_gpu_memory else None
    tok, model = load_model(args.model, device=args.device, quantize=args.quantize, max_memory=max_memory)
    st = Steerer(model, tok)
    session = Session(st, args)

    try:
        for L, n, _ in session.spec:
            st.down_col(L, n)
    except ValueError as e:
        sys.exit(f"steer: {e}")

    if args.interactive:
        if args.prompt:
            session.run(args.prompt)
        session.loop()
    else:
        session.run(args.prompt or DEFAULT_PROMPT)


if __name__ == "__main__":
    main()
