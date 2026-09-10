# Phase C — 접촉 갭 성분 분해

**접촉 안전 검증에 쓰이는 최대 접촉력은 엔진 간 7~13배 갈리고
적분 간격에 따라 최대 +676% 변한다. 반면 ISO 표준 자신이 유도에 쓰는
흡수 에너지는 세 엔진에서 1.02배 안에 들어온다.**

로봇이 사람이나 물체에 닿는 순간의 sim-to-real 갭을 성분별로 분해한다.
자유 운동 절차(PROTOCOL v0)가 다루지 않던 영역이다.

## 빠르게 확인하기

```bash
pip install -r requirements.txt
python analysis/contact/reproduce_contact.py    # 발견 1~4   약 8초
python analysis/contact/iso_criterion.py        # 발견 12    즉시
```

이 둘만 돌려도 핵심은 보인다. 나머지는 시간이 더 걸린다.

```bash
python analysis/contact/energy_portability.py   # 발견 13    약 5분
python analysis/contact/cross_engine.py         # 발견 5~7   약 15분
python analysis/contact/compliant_tip.py        # 발견 10~11 약 20분
python analysis/contact/calibrate_engines.py    # 정적 정합  약 10분
python analysis/contact/impact_calibrate.py     # 발견 8~9   약 25분
```

하드웨어 불필요. 레포 루트에서 실행한다.

## 이 단계가 무엇이 아닌지

**실물 로봇 데이터가 아니다.** 정답 모델을 실물 자리에 놓고
후보 시뮬이 얼마나 벗어나는지를 재는, 정답을 아는 상태에서의
파이프라인 검증이다. 자유 운동에서 Phase 0을 먼저 돌린 것과 같은 순서다.

## 발견 요약

| # | 내용 |
|---|---|
| 1 | 접촉 시상수를 0.006→0.020으로 무르게 잡으면 최대 충격력을 −50% 과소평가한다 (MuJoCo 내부) |
| 2 | 시뮬은 "안전", 실물은 문턱 초과인 구간이 존재한다. 책임의 96.8%가 접촉강성 하나에 있다 |
| 3 | 지배 성분 순위가 접촉 유형에 따라 통째로 바뀐다. 충돌→접촉강성, 지속압박→액추에이터 손실 |
| 4 | 궤적 RMS 갭은 센서 노이즈에 잠긴다. 같은 데이터에서 최대력은 3.5~27배 신호가 살아있다 |
| 5 | 3개 엔진 5개 설정 중 4개에서 최대 접촉력이 시간간격에 대해 발산한다 (+48~676%) |
| 6 | 충격량은 엔진 간 8~17%로 일치하고 간격에 대해서도 수렴한다. 최대력은 7~13배 벌어진다 |
| 7 | 엔진을 바꾸면 5개 조건 중 2개에서 안전 판정이 갈린다 |
| 8 | 정적 관측량으로 정합하면 충돌에서 팔이 패드를 뚫고 지나간다 (400배 밖 외삽) |
| 9 | 충돌 영역에서 정합해도 최대력이 2~8배 갈린다. 정합은 차이를 없애지 않고 옮긴다 |
| 10 | 탄성 접촉부는 안전 한계속도를 3.2배 올리고 환경 민감도를 4.69→2.2배로 낮춘다 |
| 11 | 그러나 시뮬 신뢰도는 안 오른다. 시뮬은 경고를 내기 전에 관통이탈로 먼저 무효가 된다 |
| 12 | ISO 표준의 해석해가 시뮬 측정과 13% 안에서 맞는다 — 시뮬레이터 없이 |
| 13 | 흡수 에너지는 엔진 간 1.02배로 일치. 표준이 쓰는 양이 가장 잘 옮겨진다 |

자세한 해석: [`findings/contact-gap-first-findings.md`](findings/contact-gap-first-findings.md)

## 절차서 Phase C

자유 운동용 규칙 8개에 접촉용 규칙 14개를 더했다.
전부 실제로 겪은 실패에서 역산했다.

- 규칙 9~15: 접촉 유형 축, 중력 분리, 안전 지표 분리, 과소/과대 구분,
  거짓 안전 탐색, 피크 아티팩트 검증, 지표별 노이즈 바닥
- 규칙 16~18: 최대력 수렴 확인, 엔진 간 비교는 충격량, 판정 이식성 확인
- 규칙 19~22: 정합 관측량 명시, 스윕 경계 검사, 문턱 출처 명시, 준수 입증은 에너지로

[`protocol/PROTOCOL-v0-phaseC.md`](protocol/PROTOCOL-v0-phaseC.md)

## 파일

```
analysis/contact/
  contact_model.py           MuJoCo 접촉 시나리오 모델
  reproduce_contact.py       발견 1~4 재현
  cross_engine.py            발견 5~7 재현 (MuJoCo / PyBullet / Drake)
  drake_engine.py            Drake 접촉 시나리오
  calibrate_engines.py       정적 유효강성 정합
  impact_calibrate.py        충돌 영역 정합 및 3방식 비교
  calibrated_compare.py      정적 정합 후 재비교
  compliant_tip.py           직렬 탄성 접촉부 설계 검증 (발견 10~11)
  iso_criterion.py           ISO 에너지 경로 계산 (발견 12)
  energy_portability.py      흡수 에너지 이식성 (발견 13)
  analyze_contact_gap.py     성분 분해 코어
  false_safe_sweep.py        거짓 안전 구간 탐색
  safety_attribution.py      안전 판정의 성분별 책임 귀속
  regime_compare.py          접촉 유형별 순위 비교
  metric_disagreement.py     RMS 순위 vs 안전 순위 대조
findings/                    발견 해석
protocol/                    Phase C 절차서
```

## 이슈로 알려주시면 좋을 것

1. 최대 접촉력이 시간간격에 대해 발산하는 현상, 네 번째 엔진에서도 보이는지
2. 접촉 안전 검증에 충격량을 주 지표로 쓰는 사례가 있는지
   (정합 없이 비교할 때 가장 잘 맞았다는 게 반직관적이라 확인받고 싶다)
3. 실물 로봇에서 시뮬-실물 접촉력을 동시 기록한 데이터가 공개돼 있는지
