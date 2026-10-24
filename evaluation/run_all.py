"""Run enabled suites from evaluation/config.json. Default is dry-run."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from common import configure_logging, load_config, merge_runtime_flags, parse_runner_args

SUITE_SCRIPTS = {
    "retrieval": Path(__file__).parent / "retrieval" / "evaluate_retrieval.py",
    "reranker": Path(__file__).parent / "reranker" / "evaluate_reranker.py",
    "generation": Path(__file__).parent / "generation" / "evaluate_generation.py",
    "hallucination": Path(__file__).parent / "hallucination" / "evaluate_hallucination.py",
    "end_to_end": Path(__file__).parent / "end_to_end" / "evaluate_end_to_end.py",
}


def main() -> int:
    logger = configure_logging()
    args = parse_runner_args("Run all enabled Phase 2 evaluation suites")
    config = merge_runtime_flags(load_config(args.config), args)
    enabled = config.get("suites") or {}
    extra = []
    if args.execute:
        extra.append("--execute")
    extra.extend(["--config", args.config])
    if args.dataset:
        extra.extend(["--dataset", args.dataset])
    if args.output_dir:
        extra.extend(["--output-dir", args.output_dir])
    if args.max_cases is not None:
        extra.extend(["--max-cases", str(args.max_cases)])

    failures = []
    for name, script in SUITE_SCRIPTS.items():
        if not enabled.get(name, True):
            logger.info("Skipping disabled suite %s", name)
            continue
        cmd = [sys.executable, str(script), *extra]
        logger.info("Launching %s (not executed during scaffolding)", name)
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            failures.append(name)
    if failures:
        logger.error("Suites failed: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
