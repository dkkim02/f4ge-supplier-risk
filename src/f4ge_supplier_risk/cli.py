"""`sr` — 생성기 CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from f4ge_supplier_risk import config
from f4ge_supplier_risk.calibrate import run_gates
from f4ge_supplier_risk.generator.pipeline import generate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sr")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="합성 데이터 생성")
    g.add_argument("--config", default="configs/generator.yaml")
    g.add_argument("--out", default="datasets/generated")
    g.add_argument("--seed", type=int, default=None)

    sc = sub.add_parser("score", help="채점 — supplier-risk-score.v1 출력")
    sc.add_argument("--config", default="configs/generator.yaml")
    sc.add_argument("--out", default="datasets/generated/scores.jsonl")

    c = sub.add_parser("calibrate", help="검증 게이트")
    c.add_argument("--config", default="configs/generator.yaml")
    c.add_argument("--targets", default="configs/calibration-targets.yaml")
    c.add_argument("--data", default="datasets/generated")

    args = ap.parse_args(argv)

    if args.cmd == "generate":
        cfg = config.load(args.config)
        if args.seed is not None:
            cfg["seed"] = args.seed
        counts = generate(cfg, Path(config.REPO_ROOT) / args.out)
        for k, v in counts.items():
            print(f"  {k:20s} {v:>7,}")
        print(f"→ {args.out}")
        return 0

    if args.cmd == "score":
        from f4ge_supplier_risk.prediction.run import score_all, write_scores

        scored, trust, metrics = score_all(config.load(args.config))
        n = write_scores(scored, Path(config.REPO_ROOT) / args.out)
        print(
            f"  채점 {n:,}건  ·  순위상관 {metrics['rank_corr_true']:+.3f}"
            f"  ·  예상 중앙 {metrics['pred_ppm_median']:.0f} PPM"
        )
        for a, k in scored["recommended_action"].value_counts().items():
            print(f"    {a:20s} {k:>4}건")
        print(f"  신뢰도 하위 공장: {', '.join(trust.head(3).index)}")
        print(f"→ {args.out}")
        return 0

    ok = run_gates(
        config.load(args.config), config.load(args.targets), Path(config.REPO_ROOT) / args.data
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
