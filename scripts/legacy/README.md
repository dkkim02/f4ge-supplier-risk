# legacy — 공장 유형(MES 연동/미연동) 전제의 측정 스크립트

2026-09-08 저녁, FactoryOS 소스 4개(MES·CellOS·ERP·포지 기록)가 12곳 전부에서 온다는 전제로 바뀌면서
`factory_types`·`has_mes` 구분이 사라졌다. 이 스크립트들은 그 전제에서만 돈다 — 기록용으로 남긴다.

## 2026-09-10 추가 — 보고 편향 전제의 측정 스크립트

`configs/generator.yaml` 이 보고 편향을 **통제**(편향 0)하면서 편향에 기대던 측정이 뜻을 잃었다.
편향 심한 공장이 0곳이라 기저가 0% 이고 정직도 Spearman 은 nan 이 된다.

- `discrepancy_precision.py` — review_needed 의 편향 공장 적중
- `corr_axis_discrepancy.py` — bias_capability_corr × 불일치 정밀도 (CTO 질문 22 판정에 썼다)
- `floor_scenarios.py` — MES 입력 정직도 하한 × 공장 수. `mes_input_bias_range` 가 `[1.0, 1.0]` 이라 축이 무효

내린 근거는 뜻이 없어져서만이 아니다 — **통제 전에도 기저 이하였다.**
09-10 측정: review_needed 의 편향 공장 적중 47.6% vs 기저 49.9%, corr 를 -0.85 까지
올려도 43.6%. 불일치 축은 편향 공장 판별에 기여가 없었다.

되돌리려면 `tests/conftest.py` 의 `REALISTIC_BIAS` 를 설정에 얹어 돌린다.
- `review_threshold_sweep.py` — 세 임계 중 둘이 불일치 축이고 `prec_biased` 는 편향 공장을 요구한다

⚠ 남은 임계 `_RISK_TIGHTEN_Q`(예측 위험 상위 10%) 하나를 고를 sweep 이 없다. 새로 써야 한다.
