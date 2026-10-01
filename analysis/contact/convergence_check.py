"""규칙 16 개정안 검증 — 시간간격 수렴 판정을 '관측 차수 + 폭'으로.

기존 규칙 16: dt를 8배 촘촘히(0.4 → 0.05 ms) 해서 변화율이 10%를 넘으면 발산.
개정안 (SimDiff 자기 불일치, simdiff-check 0.1과 같은 판정): dt, dt/2, dt/4 세 값만으로
  믿음  : 관측 차수 log2(|y1-y2|/|y2-y3|) 가 이론 차수 ± 0.5 안 → 리처드슨 추정치를 오차로 보고
  발산  : 차이가 dt를 줄일수록 커지고 값이 한 방향으로 움직이며, 세 값의 폭이 값의 10% 초과
  폭    : 그 외. 세 값의 폭을 오차 범위로 보고
질문
  1. 5개 엔진 설정 x 접근 속도 5개 x 지표 2개(최대력, 충격량) = 50개 경우에서
     개정안(4배)이 기존 규칙(8배)과 같은 판정을 내리는가?
  2. '폭' 판정을 받은 값에서, 한 단계 더 촘촘한 dt/8 값이 그 폭 안에 드는가? (표본 외 확인)

python analysis/contact/convergence_check.py   (레포 루트에서, 약 2분)
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import cross_engine as X  # noqa: E402

DTS = X.DTS                      # [0.4, 0.2, 0.1, 0.05] ms
TOL, REL_TOL = 0.5, 0.10
ORDER = {"MuJoCo": 4}            # contact_model.py 는 RK4, 나머지는 1차


def verdict(y, order):
    """y = [y(dt), y(dt/2), y(dt/4)] → (판정, 관측차수, 폭, 상대폭)."""
    y = np.asarray(y, dtype=float)
    d1, d2 = abs(y[0] - y[1]), abs(y[1] - y[2])
    spread = float(y.max() - y.min())
    rel = spread / max(abs(y).max(), 1e-300)
    if d1 <= 1e-10 * max(1.0, abs(y[0])):
        return "믿음", float("inf"), spread, rel
    p = math.log2(d1 / d2) if d2 > 0 else float("inf")
    if abs(p - order) <= TOL:
        return "믿음", p, spread, rel
    monotone = (y[1] - y[0]) * (y[2] - y[1]) > 0
    if d2 > d1 and monotone and rel > REL_TOL:
        return "발산", p, spread, rel
    return "폭", p, spread, rel


def criteria(y, order):
    """세 판정 기준의 '보고 가능' 여부 + 참조."""
    v, p, spread, rel3 = verdict(y[:3], order)
    rel4 = (max(y) - min(y)) / max(abs(v) for v in y)
    return {
        "A_simdiff": bool(v != "발산"),                      # 사전 등록: SimDiff 관측 차수 + 발산 판정 (dt/4까지)
        "B_rule16": bool(abs(y[3] / y[0] - 1) < REL_TOL),    # 기존 규칙 16: 양 끝 변화율 (dt/8까지)
        "C_spread4": bool(rel3 <= REL_TOL),                  # 개정안: 세 값의 폭 (dt/4까지)
        "R_spread8": bool(rel4 <= REL_TOL),                  # 참조: 네 값 전부의 폭 (dt/8까지, 가장 많은 정보)
        "verdict_A": v, "p_obs": float(p), "rel3": float(rel3), "rel4": float(rel4),
    }


def run(v0s, tag):
    rows = []
    for name, fn in X.ENGINES:
        for v0 in v0s:
            vals = [(lambda r: (r[0], X.impulse(r[1], dt)))(fn(v0, dt)) for dt in DTS]
            for k, metric in enumerate(["최대력", "충격량"]):
                y = [v[k] for v in vals]
                c = criteria(y, ORDER.get(name, 1))
                rows.append(dict(set=tag, engine=name, v0=v0, metric=metric, values=y, **c))
    return rows


def score(rows):
    out = {"n": len(rows)}
    for key in ["A_simdiff", "B_rule16", "C_spread4"]:
        agree = sum(r[key] == r["R_spread8"] for r in rows)
        lenient = sum(r[key] and not r["R_spread8"] for r in rows)   # 보고 가능이라 했는데 참조는 불가 (위험한 쪽)
        strict = sum((not r[key]) and r["R_spread8"] for r in rows)
        out[key] = {"agree_with_reference": agree, "too_lenient": lenient, "too_strict": strict}
    out["A_vs_B_agree"] = sum(r["A_simdiff"] == r["B_rule16"] for r in rows)
    out["C_vs_B_agree"] = sum(r["C_spread4"] == r["B_rule16"] for r in rows)
    return out


def main():
    train = run(X.V0S, "처음 조건")                       # 개정안(C)을 떠올린 데이터
    hold = run([4.5, 7.5, 10.5, 13.5], "새 조건")          # C를 정한 뒤 처음 돌린 속도
    res = {"criteria": {
        "A_simdiff": "SimDiff 관측 차수(±0.5) + 발산(차이 증가·단조·폭>10%) — 사전 등록, dt/4까지",
        "B_rule16": "기존 규칙 16: |y(dt/8)/y(dt) - 1| < 10% — dt/8까지",
        "C_spread4": "개정안: (최대-최소)/최대 ≤ 10%, dt, dt/2, dt/4 — dt/4까지 (계산량 약 절반)",
        "R_spread8": "참조: 네 값 전부의 폭 ≤ 10% — dt/8까지"},
        "train": score(train), "holdout": score(hold)}
    for r in train + hold:
        flags = "".join("O" if r[k] else "x" for k in ["A_simdiff", "B_rule16", "C_spread4", "R_spread8"])
        print(f"{r['set']:5s} {r['engine']:18s} v0={r['v0']:4.1f} {r['metric']:4s} A B C R = {flags}  "
              f"(A판정 {r['verdict_A']}, p={r['p_obs']:5.2f}, 폭3 {r['rel3'] * 100:5.1f}%, 폭4 {r['rel4'] * 100:5.1f}%)")
    print("\n" + json.dumps(res, ensure_ascii=False, indent=1))
    out = os.path.join(os.path.dirname(__file__), "..", "..", "findings", "convergence_check.json")
    with open(out, "w") as fh:
        json.dump({**res, "rows": train + hold}, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
