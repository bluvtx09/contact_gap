"""접촉 구간의 sim-to-real 갭을 성분별로 분해한다.

자유 운동용 analyze_gap.py와 다른 점:
  - 갭의 기준량이 관절 각도가 아니라 접촉력이다 (안전 판정의 대상이므로)
  - 성분 목록에 접촉 고유 항 4개가 추가된다
  - 지표에 최대 충격력·접촉 지속시간·안전 문턱 초과가 들어간다
"""

import numpy as np
from contact_model import build, rollout, excitation, EXCITATIONS

# 실물(정답) 파라미터. 검증 단계에서는 이 값을 알고 있다고 가정한다.
TRUE = dict(
    pad_solref="0.006 1.0",       # 실물 접촉은 딱딱하다
    pad_solimp="0.95 0.99 0.001",
    mu=0.55,
    jfric=0.010,
)
TRUE_TORQUE_SCALE = 0.82  # 액추에이터 손실: 명령 토크의 82%만 관절에 전달

# 후보 시뮬레이터가 틀리게 잡고 있는 값들
WRONG = dict(
    pad_solref="0.020 1.0",       # 수치 안정성 위해 접촉을 무르게 (흔한 기본값)
    pad_solimp="0.90 0.95 0.001", # 관통 허용폭을 넓게
    mu=0.90,                      # 마찰을 실제보다 크게 봄
    jfric=0.002,                  # 관절 건마찰을 거의 무시
)
WRONG_TORQUE_SCALE = 1.00        # 명령 토크가 그대로 전달된다고 가정

COMPONENTS = {
    "C1_접촉강성": ["pad_solref"],
    "C2_관통처리": ["pad_solimp"],
    "C3_마찰계수": ["mu"],
    "C4_관절건마찰": ["jfric"],
    "C5_액추에이터손실": ["__torque__"],
}


def run(params, torque_scale, kind, seed=None, noise=0.0):
    m = build(**params)
    r = rollout(m, excitation(kind), torque_scale=torque_scale)
    if noise > 0:
        rng = np.random.default_rng(seed)
        r["f"] = r["f"] + rng.normal(0, noise, r["f"].shape)
    return r


def gap(a, b):
    """접촉력 궤적 사이의 RMS 갭 [N]."""
    return float(np.sqrt(np.mean((a["f"] - b["f"]) ** 2)))


def safety_metrics(real, sim, limit_N):
    fr, fs = real["f"], sim["f"]
    on_r, on_s = fr > 1.0, fs > 1.0
    return dict(
        peak_real=float(fr.max()),
        peak_sim=float(fs.max()),
        peak_err_pct=float((fs.max() - fr.max()) / fr.max() * 100),
        dur_real_ms=int(on_r.sum()),
        dur_sim_ms=int(on_s.sum()),
        exceed_real_ms=int((fr > limit_N).sum()),
        exceed_sim_ms=int((fs > limit_N).sum()),
        limit_N=limit_N,
    )


def decompose(kind, noise=0.0, n_rep=1):
    """한 성분씩 정답으로 되돌려, 갭이 얼마나 줄어드는지를 기여도로 삼는다."""
    real = run(TRUE, TRUE_TORQUE_SCALE, kind)
    base_sim = run(WRONG, WRONG_TORQUE_SCALE, kind)
    g0 = gap(real, base_sim)

    contrib = {}
    for name, keys in COMPONENTS.items():
        p = dict(WRONG)
        ts = WRONG_TORQUE_SCALE
        for k in keys:
            if k == "__torque__":
                ts = TRUE_TORQUE_SCALE
            else:
                p[k] = TRUE[k]
        fixed = run(p, ts, kind)
        contrib[name] = g0 - gap(real, fixed)

    # 유의성 문턱: 반복 시행 간 노이즈로 인한 갭 변동의 2배
    thr = 0.0
    if noise > 0 and n_rep > 1:
        # 같은 실물을 두 번 잰 것 사이의 갭 = 노이즈 바닥.
        # 성분 기여도가 이 바닥보다 작으면 구분 불가로 본다.
        gs = [gap(run(TRUE, TRUE_TORQUE_SCALE, kind, seed=s, noise=noise),
                  run(TRUE, TRUE_TORQUE_SCALE, kind, seed=1000 + s, noise=noise))
              for s in range(n_rep)]
        thr = float(np.mean(gs))

    return dict(total_gap=g0, contrib=contrib, threshold=thr,
                real=real, sim=base_sim)


def main():
    print("=" * 74)
    print("접촉 갭 성분 분해 — 정답을 아는 상태에서의 검증")
    print("=" * 74)
    print(f"\n정답 대비 후보 시뮬의 오설정:")
    print(f"  접촉 시상수 {TRUE['pad_solref'].split()[0]} → {WRONG['pad_solref'].split()[0]} (더 무르게)")
    print(f"  마찰계수     {TRUE['mu']} → {WRONG['mu']}")
    print(f"  관절 건마찰  {TRUE['jfric']} → {WRONG['jfric']}")
    print(f"  토크 전달률  {TRUE_TORQUE_SCALE} → {WRONG_TORQUE_SCALE}")

    ranks = {}
    for kind in EXCITATIONS:
        d = decompose(kind, noise=0.4, n_rep=8)
        print(f"\n{'-' * 74}")
        print(f"[{kind}]  전체 갭 = {d['total_gap']:.2f} N (RMS)   "
              f"유의성 문턱 = {d['threshold']:.2f} N")
        order = sorted(d["contrib"].items(), key=lambda x: -abs(x[1]))
        ranks[kind] = [n for n, _ in order]
        for i, (n, v) in enumerate(order, 1):
            share = v / d["total_gap"] * 100
            mark = "" if abs(v) > d["threshold"] else "  (문턱 이하)"
            print(f"  {i}. {n:22s} {v:+8.2f} N  ({share:+6.1f}%){mark}")

        s = safety_metrics(d["real"], d["sim"], limit_N=140.0)
        print(f"  안전 지표 (문턱 {s['limit_N']:.0f} N):")
        print(f"    최대 충격력  실물 {s['peak_real']:6.1f} N / 시뮬 {s['peak_sim']:6.1f} N"
              f"   → 오차 {s['peak_err_pct']:+.1f}%")
        print(f"    문턱 초과시간 실물 {s['exceed_real_ms']:4d} ms / 시뮬 {s['exceed_sim_ms']:4d} ms")

    print(f"\n{'=' * 74}")
    print("궤적별 지배 성분 순위")
    print("=" * 74)
    for k, v in ranks.items():
        print(f"  {k:12s}: {' > '.join(x.split('_')[0] for x in v)}")
    uniq = {tuple(v) for v in ranks.values()}
    print(f"\n  서로 다른 순위 {len(uniq)}종 / 궤적 {len(ranks)}종")


if __name__ == "__main__":
    main()
