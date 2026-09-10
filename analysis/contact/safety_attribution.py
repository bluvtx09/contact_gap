"""거짓 안전 판정의 책임을 성분별로 귀속한다.

RMS 갭 기여도와 별개로, "안전 판정을 뒤집는 힘"이 어느 성분에
있는지를 따로 잰다. 둘은 같지 않을 수 있고, 안전 리포트에
써야 하는 건 후자다.
"""

import numpy as np
from contact_model import build, rollout
from analyze_contact_gap import (TRUE, WRONG, TRUE_TORQUE_SCALE,
                                 WRONG_TORQUE_SCALE, COMPONENTS)
from false_safe_sweep import Q_START, HOLD_TAU, LIMIT_N


def peak(params, torque_scale, v0):
    m = build(**params)
    r = rollout(m, lambda t: HOLD_TAU, torque_scale=torque_scale,
                dt_out=0.0002, q0=Q_START, v0=v0)
    return float(r["f"].max())


def attribute(v0):
    p_real = peak(TRUE, TRUE_TORQUE_SCALE, v0)
    p_sim = peak(WRONG, WRONG_TORQUE_SCALE, v0)
    err0 = p_sim - p_real

    rows = []
    for name, keys in COMPONENTS.items():
        p = dict(WRONG)
        ts = WRONG_TORQUE_SCALE
        for k in keys:
            if k == "__torque__":
                ts = TRUE_TORQUE_SCALE
            else:
                p[k] = TRUE[k]
        p_fix = peak(p, ts, v0)
        closed = abs(err0) - abs(p_fix - p_real)
        rows.append((name, p_fix, closed, closed / abs(err0) * 100))
    return p_real, p_sim, err0, sorted(rows, key=lambda r: -r[2])


def main():
    print("=" * 76)
    print("거짓 안전 판정의 성분별 책임")
    print(f"안전 문턱 {LIMIT_N:.0f} N")
    print("=" * 76)

    for v0 in [9.0, 12.0, 15.0]:
        p_real, p_sim, err0, rows = attribute(v0)
        print(f"\n{'-' * 76}")
        print(f"접근속도 {v0:.1f} rad/s")
        print(f"  실물 {p_real:.1f} N (문턱 {p_real - LIMIT_N:+.1f} N)"
              f"   시뮬 {p_sim:.1f} N (문턱 {p_sim - LIMIT_N:+.1f} N)"
              f"   과소평가 {err0:+.1f} N")
        print(f"\n  {'성분':<20}{'교정 후 시뮬':>12}{'메운 오차':>12}{'책임 몫':>10}  판정")
        for name, p_fix, closed, share in rows:
            flips = "→ 초과로 뒤집힘" if p_fix > LIMIT_N else ""
            print(f"  {name:<20}{p_fix:10.1f} N{closed:+10.1f} N{share:+9.1f}%  {flips}")

        fixers = [n for n, pf, _, _ in rows if pf > LIMIT_N]
        if fixers:
            print(f"\n  단독 교정만으로 판정이 뒤집히는 성분: {', '.join(fixers)}")
        else:
            print(f"\n  단독 교정으로는 어느 성분도 판정을 뒤집지 못한다 "
                  f"(성분 조합이 필요하다)")


if __name__ == "__main__":
    main()
