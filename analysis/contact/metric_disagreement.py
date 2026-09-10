"""같은 조건에서 두 지표의 성분 순위를 나란히 비교한다.

지표 A: 접촉력 궤적의 RMS 갭      — "시뮬이 평균적으로 얼마나 맞나"
지표 B: 최대 충격력 오차           — "안전 판정이 뒤집히나"

두 순위가 다르다면, RMS를 줄이는 방향의 시뮬 튜닝은
안전 판정을 고쳐주지 못한다.
"""

import numpy as np
from contact_model import build, rollout
from analyze_contact_gap import (TRUE, WRONG, TRUE_TORQUE_SCALE,
                                 WRONG_TORQUE_SCALE, COMPONENTS)
from false_safe_sweep import Q_START, HOLD_TAU, LIMIT_N

V0S = [6.0, 9.0, 12.0, 15.0]


def trace(params, torque_scale, v0):
    m = build(**params)
    return rollout(m, lambda t: HOLD_TAU, torque_scale=torque_scale,
                   dt_out=0.0002, q0=Q_START, v0=v0)


def both_rankings(v0):
    real = trace(TRUE, TRUE_TORQUE_SCALE, v0)
    sim = trace(WRONG, WRONG_TORQUE_SCALE, v0)

    rms0 = np.sqrt(np.mean((real["f"] - sim["f"]) ** 2))
    pk0 = abs(sim["f"].max() - real["f"].max())

    rms_c, pk_c = {}, {}
    for name, keys in COMPONENTS.items():
        p, ts = dict(WRONG), WRONG_TORQUE_SCALE
        for k in keys:
            if k == "__torque__":
                ts = TRUE_TORQUE_SCALE
            else:
                p[k] = TRUE[k]
        fx = trace(p, ts, v0)
        rms_c[name] = rms0 - np.sqrt(np.mean((real["f"] - fx["f"]) ** 2))
        pk_c[name] = pk0 - abs(fx["f"].max() - real["f"].max())

    rk_rms = [n for n, _ in sorted(rms_c.items(), key=lambda x: -abs(x[1]))]
    rk_pk = [n for n, _ in sorted(pk_c.items(), key=lambda x: -abs(x[1]))]
    return rms0, pk0, rms_c, pk_c, rk_rms, rk_pk


def short(n):
    return n.split("_")[0]


def main():
    print("=" * 78)
    print("지표 불일치 검증 — RMS 갭 순위 vs 안전 판정 책임 순위")
    print("=" * 78)

    agree_top = 0
    for v0 in V0S:
        rms0, pk0, rms_c, pk_c, rk_rms, rk_pk = both_rankings(v0)
        print(f"\n{'-' * 78}")
        print(f"접근속도 {v0:.1f} rad/s   RMS 갭 {rms0:.2f} N   최대힘 오차 {pk0:.1f} N")
        print(f"  {'성분':<20}{'RMS 기여':>12}{'  ':>2}{'최대힘 기여':>12}")
        for n in COMPONENTS:
            print(f"  {n:<20}{rms_c[n]:+10.2f} N  {pk_c[n]:+10.1f} N")
        print(f"  RMS 순위    : {' > '.join(short(x) for x in rk_rms)}")
        print(f"  안전 책임순위: {' > '.join(short(x) for x in rk_pk)}")
        same = rk_rms[0] == rk_pk[0]
        agree_top += same
        print(f"  1위 일치? {'예' if same else '아니오'}"
              f"   (RMS 1위={short(rk_rms[0])}, 안전 1위={short(rk_pk[0])})")

    print(f"\n{'=' * 78}")
    print(f"1위 성분이 일치한 조건: {agree_top}/{len(V0S)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
