# 계약 스키마 — 미작성

⚠ **작성 전에 확인이 필요하다** (`메모/일지/2026-09-08_CTO질문.md` E-18):
`quality-prediction.v1` 은 FactoryOS 정본·수정 금지이고 `score_type: "quality_fail"` 고정,
추가 필드 금지라서 **오더 단위 예측이 들어가지 않는다.**
`cell-anomaly-signal.v1` 처럼 우리가 새 계약을 정의해도 되는지 확인 필요.

## 계획된 스키마

| 파일 | 역할 | 대응 생성 파일 |
| --- | --- | --- |
| `supplier-order.v1` | 입력 — 오더 (발주 시점 확정) | `orders.jsonl` |
| `factory-report.v1` | 입력 — 공장 주간 보고 (T1+T2) | `factory_reports.jsonl` |
| `fai-report.v1` | 입력 — 초도품 검사 | `fai_reports.jsonl` |
| `order-quality-outcome.v1` | 라벨 — 입고검사·클레임 | `quality_outcomes.jsonl` |
| `supplier-risk-score.v1` | **출력** — 예측 품질 + 보고 품질 + 불일치도 | — |

## 승계하는 절대 규칙

ISO-8601 UTC 문자열(epoch ms 금지) · ID 전부 소문자 F4GE prefix ·
`part_id`/`product_id`/`operation_id` 금지 → `step_no` + `process_code` ·
수량은 `good_quantity`/`scrap_quantity` · Ground Truth 는 별도 파일 · `tenant_id`/`factory_id` 필수
