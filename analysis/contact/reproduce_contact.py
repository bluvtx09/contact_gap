"""접촉 갭 분해 — 발견 4개를 한 번에 재현한다.

    python analysis/contact/reproduce_contact.py

하드웨어 불필요. MuJoCo만 있으면 1분 안에 끝난다.

주의: 이 단계는 '정답을 아는 상태에서의 검증'이다.
실물 로봇 데이터가 아니라, 정답 모델을 실물 자리에 놓고
후보 시뮬이 그 정답에서 얼마나 벗어나는지를 잰다.
파이프라인이 성분을 제대로 되찾는지 확인하는 것이 목적이며,
실물 로봇에 대한 주장이 아니다.
"""

import numpy as np
from contact_model import build, rollout
from analyze_contact_gap import (TRUE, WRONG, TRUE_TORQUE_SCALE,
                                 WRONG_TORQUE_SCALE, COMPONENTS)

Q_START = 0.1250
LIMIT_N = 140.0
NOISE = 1.5

REGIMES = {
    "충돌_저속": dict(v0=6.0, tau=0.30),
    "충돌_고속": dict(v0=12.0, tau=0.30),
    "압박_약": dict(v0=0.0, tau=2.5),
    "압박_강": dict(v0=0.0, tau=7.0),
}


def trace(params, ts, rg):
    m = build(**params)
    return rollout(m, lambda t: rg["tau"], torque_scale=ts,
                   dt_out=0.0002, q0=Q_START, v0=rg["v0"])


def components(rg):
    real = trace(TRUE, TRUE_TORQUE_SCALE, rg)
    sim = trace(WRONG, WRONG_TORQUE_SCALE, rg)
    rms0 = np.sqrt(np.mean((real["f"] - sim["f"]) ** 2))
    pk0 = abs(sim["f"].max() - real["f"].max())
    rms_c, pk_c = {}, {}
    for n, keys in COMPONENTS.items():
        p, ts = dict(WRONG), WRONG_TORQUE_SCALE
        for k in keys:
            if k == "__torque__":
                ts = TRUE_TORQUE_SCALE
            else:
                p[k] = TRUE[k]
        fx = trace(p, ts, rg)
        rms_c[n] = rms0 - np.sqrt(np.mean((real["f"] - fx["f"]) ** 2))
        pk_c[n] = pk0 - abs(fx["f"].max() - real["f"].max())
    return real, sim, rms0, pk0, rms_c, pk_c


def s(n):
    return n.split("_")[0]


def hdr(i, t):
    print(f"\n{'=' * 76}\n발견 {i}. {t}\n{'=' * 76}")


def f1():
    hdr(1, "후보 시뮬이 최대 충격력을 절반 수준으로 과소평가한다")
    print(f"\n{'접근속도':>8}{'실물 peak':>11}{'시뮬 peak':>11}{'오차':>9}")
    print("-" * 42)
    for v0 in [3.0, 6.0, 9.0, 12.0, 15.0]:
        rg = dict(v0=v0, tau=0.30)
        pr = trace(TRUE, TRUE_TORQUE_SCALE, rg)["f"].max()
        ps = trace(WRONG, WRONG_TORQUE_SCALE, rg)["f"].max()
        print(f"{v0:8.1f}{pr:10.1f} N{ps:10.1f} N{(ps - pr) / pr * 100:+8.1f}%")
    print("\n과소평가는 안전 판정에서 위험한 방향이다.")


def f2():
    hdr(2, f"시뮬이 '안전'이라 판정하는데 실물은 문턱을 넘는 구간이 존재한다")
    print(f"\n안전 문턱 {LIMIT_N:.0f} N")
    print(f"\n{'접근속도':>8}{'실물':>10}{'시뮬':>10}  {'시뮬판정':>8}{'실물결과':>8}  판정")
    print("-" * 60)
    fs = []
    lv = np.arange(5.0, 15.1, 1.0)
    for v0 in lv:
        rg = dict(v0=v0, tau=0.30)
        pr = trace(TRUE, TRUE_TORQUE_SCALE, rg)["f"].max()
        ps = trace(WRONG, WRONG_TORQUE_SCALE, rg)["f"].max()
        so, ro = ps <= LIMIT_N, pr <= LIMIT_N
        v = "*** 거짓 안전 ***" if (so and not ro) else "일치"
        if so and not ro:
            fs.append(v0)
        print(f"{v0:8.1f}{pr:9.1f}N{ps:9.1f}N  {'안전' if so else '초과':>8}"
              f"{'안전' if ro else '초과':>8}  {v}")
    print(f"\n거짓 안전 경계: 실물은 {min(fs):.0f} rad/s에서 문턱을 넘지만,")
    print(f"시뮬은 시험 범위({lv[-1]:.0f} rad/s)까지 끝내 넘지 않는다.")


def f3():
    hdr(3, "지배 성분의 순위가 접촉 유형에 따라 통째로 바뀐다")
    print(f"\n{'유형':<10}{'실물peak':>10}{'시뮬peak':>10}   RMS순위 / 안전순위")
    print("-" * 66)
    rr, pr_ = {}, {}
    for name, rg in REGIMES.items():
        real, sim, rms0, pk0, rms_c, pk_c = components(rg)
        a = tuple(sorted(rms_c, key=lambda x: -abs(rms_c[x])))
        b = tuple(sorted(pk_c, key=lambda x: -abs(pk_c[x])))
        rr[name], pr_[name] = a, b
        print(f"{name:<10}{real['f'].max():9.1f}N{sim['f'].max():9.1f}N"
              f"   {' > '.join(s(x) for x in a[:3])} / {' > '.join(s(x) for x in b[:3])}")
    print(f"\nRMS 기준 서로 다른 순위 {len(set(rr.values()))}종 / {len(REGIMES)}조건")
    print(f"안전 기준 서로 다른 순위 {len(set(pr_.values()))}종 / {len(REGIMES)}조건")
    print("\n충돌에서는 접촉강성(C1)이, 지속 압박에서는 액추에이터 손실(C5)이 지배한다.")
    print("한 조건에서 잰 순위를 다른 조건에 그대로 쓸 수 없다.")


def f4():
    hdr(4, "RMS 갭으로는 성분을 잴 수 없다 — 센서 노이즈에 잠긴다")
    rng = np.random.default_rng(1)
    print(f"\n힘 센서 노이즈 {NOISE} N 가정")
    print(f"\n{'유형':<10}{'RMS기여':>9}{'RMS바닥':>9}{'배수':>8}"
          f"{'최대힘오차':>11}{'최대힘바닥':>11}{'배수':>8}")
    print("-" * 70)
    for name, rg in REGIMES.items():
        real, sim, rms0, pk0, rms_c, pk_c = components(rg)
        top = max(rms_c, key=lambda x: abs(rms_c[x]))
        rf = np.mean([np.sqrt(np.mean((rng.normal(0, NOISE, real["f"].shape)
                                       - rng.normal(0, NOISE, real["f"].shape)) ** 2))
                      for _ in range(30)])
        pd = [abs((real["f"] + rng.normal(0, NOISE, real["f"].shape)).max()
                  - (real["f"] + rng.normal(0, NOISE, real["f"].shape)).max())
              for _ in range(200)]
        pf = 2 * np.std(pd) + np.mean(pd)
        print(f"{name:<10}{rms_c[top]:+8.2f}N{rf:8.2f}N{abs(rms_c[top]) / rf:7.2f}x"
              f"{pk0:10.1f}N{pf:10.2f}N{pk0 / pf:7.1f}x")
    print("\n배수 1.0 미만이면 그 지표로는 성분을 구분할 수 없다.")
    print("RMS는 1위 성분조차 절반의 조건에서 노이즈에 잠긴다.")
    print("최대 충격력은 4개 조건 모두 노이즈 바닥의 3.5배 이상으로 살아남는다.")


def main():
    print("=" * 76)
    print("접촉 갭 성분 분해 — 정답을 아는 상태에서의 검증")
    print("=" * 76)
    print("\n후보 시뮬의 오설정:")
    print(f"  접촉 시상수  {TRUE['pad_solref'].split()[0]} → "
          f"{WRONG['pad_solref'].split()[0]}  (수치 안정성 위해 무르게)")
    print(f"  관통 허용폭  {TRUE['pad_solimp'].split()[0]} → {WRONG['pad_solimp'].split()[0]}")
    print(f"  마찰계수     {TRUE['mu']} → {WRONG['mu']}")
    print(f"  관절 건마찰  {TRUE['jfric']} → {WRONG['jfric']}")
    print(f"  토크 전달률  {TRUE_TORQUE_SCALE} → {WRONG_TORQUE_SCALE}")
    f1(); f2(); f3(); f4()
    print(f"\n{'=' * 76}")
    print("해석은 findings/contact-gap-first-findings.md 참고")
    print("=" * 76)


if __name__ == "__main__":
    main()
