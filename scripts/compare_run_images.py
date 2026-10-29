"""Compare two `python -m ocr run-images` outputs on the same folder (old code vs new code).

Works with output from the original OCR branch as well as this one, because it only reads
`summary.json`. Put `<image>.txt` (the correct text) next to each image to get CER.

Usage:
    python -m ocr run-images --input data/images --out outputs/before --variant original   # on the old branch
    python -m ocr run-images --input data/images --out outputs/after  --variant original   # on the new branch
    python scripts/compare_run_images.py outputs/before outputs/after
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(directory: Path) -> dict[str, dict]:
    payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    return {item["source"]: item for item in payload["images"]}


def compare(old: dict[str, dict], new: dict[str, dict]) -> dict:
    shared = sorted(set(old) & set(new))
    fixed = [i for i in shared if old[i]["status"] != "success" and new[i]["status"] == "success"]
    broken = [i for i in shared if old[i]["status"] == "success" and new[i]["status"] != "success"]
    both = [i for i in shared if "cer" in old[i] and "cer" in new[i]]
    better = [i for i in both if new[i]["cer"] < old[i]["cer"] - 1e-9]
    worse = [i for i in both if new[i]["cer"] > old[i]["cer"] + 1e-9]
    return {
        "images": len(shared),
        "old_ok": sum(old[i]["status"] == "success" for i in shared),
        "new_ok": sum(new[i]["status"] == "success" for i in shared),
        "fixed": fixed,
        "broken": broken,
        "scored": both,
        "better": better,
        "worse": worse,
        "mean_cer_old": sum(old[i]["cer"] for i in both) / len(both) if both else None,
        "mean_cer_new": sum(new[i]["cer"] for i in both) / len(both) if both else None,
        "only_old": sorted(set(old) - set(new)),
        "only_new": sorted(set(new) - set(old)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    args = parser.parse_args(argv)
    result = compare(load(args.old), load(args.new))

    print(f"{result['images']} images in both runs. Recognized: {result['old_ok']} -> {result['new_ok']}")
    for label, key in (("now recognized", "fixed"), ("no longer recognized", "broken")):
        if result[key]:
            print(f"\n{label}:")
            for image in result[key]:
                print(f"  {image}")
    if result["scored"]:
        print(
            f"\nCER on {len(result['scored'])} images that have ground truth and were read in both runs "
            f"(images that failed are not in this average): {result['mean_cer_old']:.3f} -> {result['mean_cer_new']:.3f}"
        )
        print(f"  better: {len(result['better'])}   worse: {len(result['worse'])}")
        for image in result["worse"]:
            print(f"  worse: {image}")
    else:
        print("\nNo CER: add <image>.txt files with the correct text next to the images.")
    for label, key in (("only in the old run", "only_old"), ("only in the new run", "only_new")):
        if result[key]:
            print(f"\n{label}: {', '.join(result[key])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
