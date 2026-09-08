"""`sr` — 생성기 · 채점 · 운영 서버 CLI."""

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

    sv = sub.add_parser("serve", help="운영 서버 (FastAPI)")
    sv.add_argument("--db", default=None, help="SQLAlchemy URL. 기본 sqlite:///data/supplier_risk.db")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8020)

    sd = sub.add_parser("seed", help="생성 데이터를 계약 경로로 DB 에 넣고 admin 키를 발급")
    sd.add_argument("--db", default=None)
    sd.add_argument("--config", default="configs/generator.yaml")
    sd.add_argument("--score", action="store_true", help="넣은 뒤 바로 채점")

    ky = sub.add_parser("key", help="API 키 발급")
    ky.add_argument("--db", default=None)
    ky.add_argument("--role", choices=("admin", "site", "ingest"), required=True)
    ky.add_argument("--label", required=True)
    ky.add_argument("--factory", default=None, help="site 키의 공장 id")

    sr_ = sub.add_parser("score-run", help="DB 의 계약 행으로 채점 1회 (cron 용)")
    sr_.add_argument("--db", default=None)

    args = ap.parse_args(argv)

    if args.cmd in ("serve", "seed", "key", "score-run"):
        from f4ge_supplier_risk.web import auth, db

        eng = db.connect(args.db or db.DEFAULT_URL)
        if args.cmd == "serve":
            import uvicorn

            from f4ge_supplier_risk.web.service import create_app

            uvicorn.run(create_app(eng), host=args.host, port=args.port)
            return 0
        if args.cmd == "seed":
            from f4ge_supplier_risk.generator.pipeline import build_dataset
            from f4ge_supplier_risk.web.service import seed_from_generated

            counts = seed_from_generated(eng, build_dataset(config.load(args.config)))
            for k, v in counts.items():
                print(f"  {k:28s} {v:>7,}")
            key = auth.issue_key(eng, "seed-admin", "admin")
            print(f"  admin 키 (한 번만 표시): {key}")
            if args.score:
                from f4ge_supplier_risk.prediction.live import score_from_db

                print(" ", score_from_db(eng, note="seed"))
            return 0
        if args.cmd == "key":
            print(auth.issue_key(eng, args.label, args.role, args.factory))
            return 0
        from f4ge_supplier_risk.prediction.live import score_from_db

        print(score_from_db(eng, note="cli"))
        return 0

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
            f"  채점 {n:,}건  ·  Spearman 상관 {metrics['rank_corr_true']:+.3f}"
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
