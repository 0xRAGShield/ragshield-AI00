"""Compare two `python -m ocr triage` runs on the same folder (for example old code vs new code).

Usage:
    python scripts/compare_triage.py outputs/triage_old outputs/triage_new

Prints which images changed bucket, how line counts and CER moved, and a one-line verdict.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ORDER = {"ok": 0, "recognition": 1, "input_quality": 1, "detection": 2, "unreadable_file": 3}


def load(directory: Path) -> dict[str, dict]:
    with (directory / "triage.csv").open(encoding="utf-8-sig", newline="") as handle:
        return {row["image"]: row for row in csv.DictReader(handle)}


def _num(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare(old: dict[str, dict], new: dict[str, dict]) -> dict:
    improved, worsened, same, lines = [], [], 0, []
    cer_pairs = []
    for image in sorted(set(old) & set(new)):
        before, after = old[image], new[image]
        rank_before, rank_after = ORDER.get(before["bucket"], 1), ORDER.get(after["bucket"], 1)
        if rank_after < rank_before:
            improved.append((image, before["bucket"], after["bucket"]))
        elif rank_after > rank_before:
            worsened.append((image, before["bucket"], after["bucket"]))
        else:
            same += 1
        l0, l1 = _num(before["lines"]), _num(after["lines"])
        if l0 is not None and l1 is not None and l0 != l1:
            lines.append((image, int(l0), int(l1)))
        c0, c1 = _num(before.get("cer", "")), _num(after.get("cer", ""))
        if c0 is not None and c1 is not None:
            cer_pairs.append((image, c0, c1))
    return {
        "improved": improved,
        "worsened": worsened,
        "unchanged": same,
        "line_changes": lines,
        "cer": cer_pairs,
        "only_old": sorted(set(old) - set(new)),
        "only_new": sorted(set(new) - set(old)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    args = parser.parse_args(argv)
    result = compare(load(args.old), load(args.new))

    print(f"improved: {len(result['improved'])}   worsened: {len(result['worsened'])}   unchanged: {result['unchanged']}")
    for image, a, b in result["improved"]:
        print(f"  + {image}: {a} -> {b}")
    for image, a, b in result["worsened"]:
        print(f"  - {image}: {a} -> {b}")
    if result["line_changes"]:
        print("\nline count changes (old -> new):")
        for image, a, b in result["line_changes"]:
            print(f"  {image}: {a} -> {b}")
    if result["cer"]:
        mean_old = sum(c0 for _, c0, _ in result["cer"]) / len(result["cer"])
        mean_new = sum(c1 for _, _, c1 in result["cer"]) / len(result["cer"])
        print(f"\nmean CER over {len(result['cer'])} images with ground truth: {mean_old:.3f} -> {mean_new:.3f}")
    for label, key in (("only in old run", "only_old"), ("only in new run", "only_new")):
        if result[key]:
            print(f"\n{label}: {', '.join(result[key])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
