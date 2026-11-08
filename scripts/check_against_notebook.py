"""End-to-end check of the real Qwen3.5-4B against the notebook's saved (CUDA, 4-bit NF4) outputs.

    uv run scripts/check_against_notebook.py                 # default precision for the device
    uv run scripts/check_against_notebook.py --quantize      # 4-bit NF4 (CUDA default)
    uv run scripts/check_against_notebook.py --no-quantize   # bf16 (MPS/CPU default)

Hard checks (exit 1 if any fail): memory footprint, calibration numbers, steering changes the
output, clearing restores it exactly, and active hooks don't leak into calibration.
Soft check (reported only): how many leading words of each generation match the outputs saved
in Steering_LLMs.ipynb. Greedy decoding amplifies tiny kernel differences, so different
hardware drifts apart after a while. The same GPU type and precision should match much longer.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from steering_llms import Steerer, load_model

NOTEBOOK = Path(__file__).resolve().parent.parent / "Steering_LLMs.ipynb"
HUMANS = "What do you think about humans?"

# Expected values. NF4 numbers come from the friend's CUDA run (and agree with NF4 on MPS);
# bf16 numbers were measured on MPS. ||col|| for L14 n8949 is the true column norm: the 7.97 in
# the saved notebook was the leaked 7.5 steering hook from an earlier cell.
EXPECTED = {
    "nf4": {"footprint_gib": 2.85, "median_norm_13": 9.189, "col_13_7956": 0.4195, "col_14_8949": 0.4647},
    "bf16": {"footprint_gib": 7.83, "median_norm_13": 9.466, "col_13_7956": 0.4183, "col_14_8949": 0.4647},
}
TOLERANCE = {"footprint_gib": 0.05, "median_norm_13": 0.03, "col_13_7956": 0.02, "col_14_8949": 0.02}  # abs GiB / rel

# (source of the notebook cell holding the saved output, steering spec, prompt)
GENERATIONS = [
    ("print(try_steering([]))", [], HUMANS),
    ('print(try_steering([(14,8949,7.5)], "What do you think about humans?", max_new_tokens=100))', [(14, 8949, 7.5)], HUMANS),
    ('print(try_steering([(17,7426,5)], "What do you think about humans?", max_new_tokens=100))', [(17, 7426, 5)], HUMANS),
]


def saved_output(source: str) -> str:
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if "".join(cell["source"]).strip() == source:
            text = "".join("".join(o.get("text", "")) for o in cell.get("outputs", []) if o.get("name") == "stdout")
            # the original try_steering printed "torch.Size([2560]) tensor(...)" debug lines first
            return "\n".join(l for l in text.splitlines() if not l.startswith("torch.Size")).strip()
    raise KeyError(f"no notebook cell with source {source!r}")


def same_leading_words(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a.split(), b.split()):
        if x != y:
            break
        n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=["cuda", "mps", "cpu"])
    p.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=None, help="default: on for CUDA only")
    p.add_argument("--max-new-tokens", type=int, default=100)
    args = p.parse_args()

    t0 = time.time()
    tok, model = load_model(device=args.device, quantize=args.quantize)
    st = Steerer(model, tok)
    quantized = type(st.layers[0].mlp.down_proj).__name__ == "Linear4bit"
    variant = "nf4" if quantized else "bf16"
    exp = EXPECTED[variant]

    print(f"torch {torch.__version__}, device {model.device}, {variant}, loaded in {time.time() - t0:.1f}s")
    if model.device.type == "cuda":
        cap = torch.cuda.get_device_capability()
        print(f"GPU {torch.cuda.get_device_name()} (sm_{cap[0]}{cap[1]}), CUDA {torch.version.cuda}")

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        if not ok:
            failures.append(name)

    if args.device:
        check("device", model.device.type == args.device, f"model is on {model.device} (asked for {args.device})")
    gib = model.get_memory_footprint() / 2**30
    check("memory footprint", abs(gib - exp["footprint_gib"]) <= TOLERANCE["footprint_gib"],
          f"{gib:.2f} GiB (expected {exp['footprint_gib']})")

    measured = {
        "median_norm_13": st.median_norm(13),
        "col_13_7956": st.down_col(13, 7956).norm().item(),
        "col_14_8949": st.down_col(14, 8949).norm().item(),
    }
    for key, val in measured.items():
        rel = abs(val - exp[key]) / exp[key]
        check(key, rel <= TOLERANCE[key], f"{val:.4f} (expected {exp[key]}, off by {rel:.1%})")
    med14 = st.median_norm(14)
    print(f"[info] median_norm_14: {med14:.3f} (saved notebook's 12.75 was measured with a hook on)")

    hooked = Steerer(model, tok)  # fresh cache; hooks left on, like after notebook cell 36
    hooked.set([(14, 8949, 7.5), (17, 7426, -5)])
    col, med = hooked.down_col(14, 8949).norm().item(), hooked.median_norm(14)
    still_on = hooked.active
    hooked.clear()
    check("no hook leak", abs(col - measured["col_14_8949"]) < 1e-5 and abs(med - med14) < 1e-3 * med14 and still_on,
          f"with steering on L14: ||col|| {col:.4f}, median_norm_14 {med:.3f}")

    outputs = {}
    for source, spec, prompt in GENERATIONS:
        t1 = time.time()
        with st.steering(spec):
            out = st.generate(prompt, args.max_new_tokens).strip()
        outputs[str(spec)] = out
        ref = saved_output(source)
        k = same_leading_words(out, ref)
        print(f"\n[info] {spec or 'baseline'} ({time.time() - t1:.1f}s): first {k} words identical to the saved CUDA NF4 output")
        print("  here :", out[:240].replace("\n", " "))
        print("  saved:", ref[:240].replace("\n", " "))

    print()
    base = outputs["[]"]
    check("steering changes output", all(o != base for s, o in outputs.items() if s != "[]"), "evil and poet differ from baseline")
    again = st.generate(HUMANS, args.max_new_tokens).strip()
    check("clear restores baseline", again == base, "unsteered output identical before and after steering")

    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILED: ' + ', '.join(failures)} ({time.time() - t0:.0f}s total)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
