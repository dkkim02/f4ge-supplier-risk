# 계약 스키마 — 입력 5종 · 출력 1종 (초안, 2026-09-08)

JSON Schema 2020-12. 각 스키마는 `datasets/generated/*.jsonl` 의 한 줄에 대응하고,
`src/f4ge_supplier_risk/contracts.py` 가 생성 레코드를 계약 행으로 번역한다.
**`tests/unit/test_contracts.py` 가 생성 레코드 전부를 각 계약으로 검증한다** — 계약과 생성기가 어긋나면 테스트가 깨진다.
`examples/` 는 그 번역기로 만든 실제 행이다.

⚠ **정의 권한은 아직 확인 전이다** (`메모/일지/2026-09-08_CTO질문.md` 18번):
FactoryOS 의 `quality-prediction.v1` 은 정본·수정 금지이고 `score_type: "quality_fail"` 고정, 추가 필드 금지라서
오더 단위 예측이 들어가지 않는다. `cell-anomaly-signal.v1` 처럼 우리가 새 계약을 정의해도 되는지 확인이 필요하다.
그때까지 이 여섯 개는 **초안**이다.

| 파일 | 역할 | 오는 곳 | 대응 생성 파일 |
| --- | --- | --- | --- |
| `supplier-order.v1` | 입력 — 오더. **발주 시점에 확정되는 값만** | 우리(L0) | `orders.jsonl` |
| `factory-report.v1` | 입력 — **공장 MES 일 1회 집계**(T1+T2). `is_missing` = API 끊김 | 자체 MES → FactoryOS 연동 · **유형 a(3곳)만** | `factory_reports.jsonl` |
| `fai-report.v1` | 입력 — 초도품 검사(AS9102 Form 3 요약). 유형 무관 | 공장 / 우리 대행 / 제3자 | `fai_reports.jsonl` |
| `cell-daily.v1` | 입력 — 설비 텔레메트리 일 집계. **교차검증용** | CellOS · **12곳 전부**(협력 조건) | `cell_daily.jsonl` |
| `order-quality-outcome.v1` | **라벨** — 입고검사(표본 n 중 불합격 k) + 클레임. `label_available_at` 이 핵심 | 우리 · 고객 검사 | `quality_outcomes.jsonl` |
| `supplier-risk-score.v1` | **출력** — 예측 품질 + 보고 품질 + 불일치도 + 권고 | 모델 | `scores.jsonl` |

## 승계한 규칙과, 이 프로젝트에서 정한 것

FactoryOS 계약에서 그대로: ISO-8601 UTC 문자열(epoch ms 금지) · id 전부 소문자 prefix(`ten_` `ord_` `fac_` `mtl_` `sup_` `mch_` `cell_`) ·
수량은 `*_quantity` · Ground Truth 는 별도 파일(계약 없음, 평가 전용) · `tenant_id`/`factory_id` 필수 · `additionalProperties: false`.

이 프로젝트에서 정한 것:
- **`product_id` 금지 → `product_code`.** FactoryOS 는 `step_no + process_code` 를 쓰지만 오더는 공정이 아니라 제품군 단위다.
  같은 제품군을 여러 공장이 만드는 것(배분 모델의 전제)을 `product_code` 로 표현한다.
- **수량 셋을 분리한다** — `produced_quantity` / `scrap_quantity` / `rework_quantity`. FactoryOS 의 `good_quantity` 는 produced − scrap − rework 로 유도.
  재작업을 폐기와 합치면 문제의 3% 만 보인다(Hidden Factory).
- **결측은 값이 아니라 상태다.** `factory-report.v1` 의 `is_missing=true` 행은 수량 필드를 가질 수 없다(`if/then`). MES 미연동 공장은 행 자체가 없다.
- **`label_available_at`** — 결과를 우리가 알게 된 시각. 모델은 이 시각 이전에 그 행을 볼 수 없다. `market` 이 이 지연을 정한다(18/30/60일).
- 파생값은 계약에 넣지 않는다 — `lead_slack` `price_zscore` `load_index` 는 원값에서 피처 빌더가 만든다.
- `tenant_id` 는 생성기에 없다 — 번역기가 `ten_f4ge` 를 채운다. 실서비스에서는 수신 측이 채운다.

## 들어오는 길

`POST /api/ingest/{계약이름}` 에 행 배열을 보낸다(`X-API-Key`, 역할 ingest·admin). 스키마 위반이 하나라도 있으면 배치 전체 422.
같은 키(order_id · seq 등)의 행은 새 것으로 덮는다. → `docs/운영서버.md`

## 검증

```
pytest tests/unit/test_contracts.py     # 생성 레코드 전부 × 계약 6종
```

09-08 초안 작성 중 발견: `supplier-risk-score.v1` 파일이 09-07 부터 **JSON 으로 깨져 있었다**(설명문의 따옴표 미이스케이프).
검증 테스트가 없었기 때문이다. 이번에 고쳤고, 이제 테스트가 막는다.
