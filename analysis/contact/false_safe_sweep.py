"""거짓 안전(false-safe) 구간 탐색.

시뮬이 "안전 문턱 미만"이라 판정하는데 실물은 문턱을 넘는
접근 조건이 존재하는가? 존재한다면 그 폭은 얼마인가?
"""

import numpy as np
from contact_model import build, rollout
from analyze_contact_gap import (TRUE, WRONG, TRUE_TORQUE_SCALE,
                                 WRONG_TORQUE_SCALE)

LIMIT_N = 140.0   # 협동로봇 접촉 안전 문턱 예시값


Q_CONTACT = 0.1350   # 접촉이 시작되는 관절각 [rad]
Q_START = 0.1250     # 접촉 직전 출발점
HOLD_TAU = 0.30      # 접촉 유지용 일정 토크 [N·m]


def peak_force(params, torque_scale, v0):
    """접근 각속도 v0 [rad/s]로 패드에 진입시켰을 때의 최대 접촉력."""
    m = build(**params)
    r = rollout(m, lambda t: HOLD_TAU, torque_scale=torque_scale,
                dt_out=0.0002, q0=Q_START, v0=v0)
    return float(r["f"].max())


def main():
    print("=" * 78)
    print("거짓 안전 구간 탐색 — 시뮬 판정 vs 실물 결과")
    print(f"안전 문턱 {LIMIT_N:.0f} N")
    print("=" * 78)
    print(f"\n{'접근속도':>8} {'실물 peak':>11} {'시뮬 peak':>11} {'오차':>9}"
          f"  {'시뮬판정':>8} {'실물결과':>8}  판정")
    print("-" * 78)

    levels = np.arange(1.0, 15.1, 0.5)
    false_safe = []
    for lv in levels:
        pr = peak_force(TRUE, TRUE_TORQUE_SCALE, lv)
        ps = peak_force(WRONG, WRONG_TORQUE_SCALE, lv)
        sim_ok = ps <= LIMIT_N
        real_ok = pr <= LIMIT_N
        if sim_ok and not real_ok:
            verdict = "*** 거짓 안전 ***"
            false_safe.append(lv)
        elif (not sim_ok) and real_ok:
            verdict = "거짓 경보"
        else:
            verdict = "일치"
        err = (ps - pr) / pr * 100
        print(f"{lv:8.2f} {pr:10.1f} N {ps:10.1f} N {err:+8.1f}%"
              f"  {'안전' if sim_ok else '초과':>8} {'안전' if real_ok else '초과':>8}  {verdict}")

    print("-" * 78)
    if false_safe:
        lo, hi = min(false_safe), max(false_safe)
        print(f"\n거짓 안전 구간: 접근속도 {lo:.2f} ~ {hi:.2f} rad/s  "
              f"({len(false_safe)}/{len(levels)} 조건 = {len(false_safe)/len(levels)*100:.0f}%)")
        mid = false_safe[len(false_safe) // 2]
        pr = peak_force(TRUE, TRUE_TORQUE_SCALE, mid)
        ps = peak_force(WRONG, WRONG_TORQUE_SCALE, mid)
        print(f"구간 중앙({mid:.2f})에서 시뮬 {ps:.1f} N → '통과',"
              f" 실물 {pr:.1f} N → 문턱 {pr - LIMIT_N:+.1f} N 초과")
    else:
        print("\n이 설정에서는 거짓 안전 구간이 나타나지 않았다.")


if __name__ == "__main__":
    main()
