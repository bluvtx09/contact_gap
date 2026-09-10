"""접촉 유형을 두 번째 독립변수로 놓고 두 가지를 동시에 본다.

(1) 지배 성분의 순위가 접촉 유형에 따라 바뀌는가  (H4의 접촉판)
(2) RMS 갭과 최대힘 오차의 크기가 얼마나 어긋나는가
"""

import numpy as np
from contact_model import build, rollout
from analyze_contact_gap import (TRUE, WRONG, TRUE_TORQUE_SCALE,
                                 WRONG_TORQUE_SCALE, COMPONENTS)

Q_START = 0.1250
LIMIT_N = 140.0

REGIMES = {
    # 순간 충돌: 접촉 직전에서 빠른 접근 속도로 진입
    "충돌_저속": dict(v0=6.0, tau=0.30, q0=Q_START),
    "충돌_고속": dict(v0=12.0, tau=0.30, q0=Q_START),
    # 지속 압박: 정지 상태에서 큰 토크로 눌러 미끄러뜨림
    "압박_약": dict(v0=0.0, tau=2.5, q0=Q_START),
    "압박_강": dict(v0=0.0, tau=7.0, q0=Q_START),
}


def trace(params, torque_scale, rg):
    m = build(**params)
    return rollout(m, lambda t: rg["tau"], torque_scale=torque_scale,
                   dt_out=0.0002, q0=rg["q0"], v0=rg["v0"])


def analyze(rg):
    real = trace(TRUE, TRUE_TORQUE_SCALE, rg)
    sim = trace(WRONG, WRONG_TORQUE_SCALE, rg)
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
        fx = trace(p, ts, rg)
        rms_c[name] = rms0 - np.sqrt(np.mean((real["f"] - fx["f"]) ** 2))
        pk_c[name] = pk0 - abs(fx["f"].max() - real["f"].max())

    return dict(real=real, sim=sim, rms0=rms0, pk0=pk0,
                rms_c=rms_c, pk_c=pk_c)


def short(n):
    return n.split("_")[0]


def main():
    print("=" * 80)
    print("접촉 유형별 성분 순위 및 지표 크기 괴리")
    print("=" * 80)

    rms_ranks, pk_ranks = {}, {}
    print(f"\n{'유형':<10}{'실물peak':>10}{'시뮬peak':>10}{'RMS갭':>9}"
          f"{'최대힘오차':>11}{'배율':>7}  RMS순위 / 안전순위")
    print("-" * 80)

    for name, rg in REGIMES.items():
        d = analyze(rg)
        rk_rms = [n for n, _ in sorted(d["rms_c"].items(), key=lambda x: -abs(x[1]))]
        rk_pk = [n for n, _ in sorted(d["pk_c"].items(), key=lambda x: -abs(x[1]))]
        rms_ranks[name], pk_ranks[name] = tuple(rk_rms), tuple(rk_pk)
        ratio = d["pk0"] / d["rms0"] if d["rms0"] > 1e-9 else float("inf")
        print(f"{name:<10}{d['real']['f'].max():9.1f}N{d['sim']['f'].max():9.1f}N"
              f"{d['rms0']:8.2f}N{d['pk0']:10.1f}N{ratio:6.1f}x"
              f"  {' > '.join(short(x) for x in rk_rms[:3])}"
              f" / {' > '.join(short(x) for x in rk_pk[:3])}")

    print("-" * 80)
    print(f"\nRMS 기준 서로 다른 순위: {len(set(rms_ranks.values()))}종 / {len(REGIMES)}조건")
    print(f"안전 기준 서로 다른 순위: {len(set(pk_ranks.values()))}종 / {len(REGIMES)}조건")

    print("\n유형별 1위 성분:")
    for name in REGIMES:
        a, b = short(rms_ranks[name][0]), short(pk_ranks[name][0])
        flag = "" if a == b else "   ← 두 지표가 다른 성분을 지목"
        print(f"  {name:<10} RMS={a:<4} 안전={b:<4}{flag}")


if __name__ == "__main__":
    main()
