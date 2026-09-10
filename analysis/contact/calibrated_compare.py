"""정합 후 재비교 — 파라미터를 맞추면 엔진 차이가 사라지는가?

2차·3차 교차 검증에서는 각 엔진에 '합리적 범위'의 값을
임의로 넣었다. 벤더 입장에서는 "설정을 제대로 안 맞춰서 그렇다"고
반박할 수 있는 구멍이다.

여기서는 정적 유효강성 1e5 N/m로 세 엔진을 정합한 뒤
같은 비교를 다시 한다.
"""

import numpy as np

CAL = {"MuJoCo": 1166.0, "PyBullet": 2.286e5, "Drake": 2.001e5}
V0S = [3.0, 6.0, 9.0, 12.0, 15.0]
DTS = [0.0004, 0.0002, 0.0001, 0.00005]
LIMIT = 140.0
T_END = 0.30
HOLD_TAU = 0.30
Q_START = 0.1250
MJ_SOLIMP = "0.9 0.95 0.001"

# 정합 전 값 (3차 교차 검증에서 측정)
BEFORE_PK = {3.0: (48.7, 467.5, 148.4), 6.0: (97.5, 1186.3, 348.6),
             9.0: (150.4, 1500.7, 797.0), 12.0: (201.2, 1498.3, 834.4),
             15.0: (249.9, 3238.1, 1816.9)}
BEFORE_IM = {3.0: (0.463, 0.430, 0.466), 6.0: (0.624, 0.568, 0.614),
             9.0: (0.802, 0.693, 0.779), 12.0: (0.969, 0.831, 0.920),
             15.0: (1.138, 0.978, 1.124)}


def run_mj(v0, dt):
    import contact_model as CM
    CM.BASE["dt"] = dt
    k = CAL["MuJoCo"]
    sr = f"-{k:.6g} -{2 * np.sqrt(k * 0.45):.6g}"
    m = CM.build(pad_solref=sr, tip_solref=sr,
                 pad_solimp=MJ_SOLIMP, tip_solimp=MJ_SOLIMP)
    r = CM.rollout(m, lambda t: HOLD_TAU, torque_scale=1.0,
                   dt_out=dt, q0=Q_START, v0=v0)
    n = int(T_END / dt)
    f = r["f"][:n]
    return float(f.max()), f


def run_pb(v0, dt):
    import cross_engine as X
    k = CAL["PyBullet"]
    return X.run_pybullet(v0, contact_stiffness=k,
                          contact_damping=2 * np.sqrt(k * 0.45), dt=dt)


def run_dk(v0, dt):
    from drake_engine import run_drake
    return run_drake(v0, dt, stiff=CAL["Drake"], diss=10.0)


ENGINES = [("MuJoCo", run_mj), ("PyBullet", run_pb), ("Drake", run_dk)]


def spread(vals):
    return max(vals) / min(vals)


def main():
    print("=" * 76)
    print("정합 후 재비교 — 정적 유효강성 1e5 N/m로 세 엔진을 맞춤")
    print("=" * 76)

    # --- 검사 1: 수렴성이 회복되는가 ---
    print(f"\n{'-' * 76}")
    print("검사 1. 정합 후에도 최대력이 발산하는가 (v0 = 12 rad/s)")
    print("-" * 76)
    print(f"\n{'엔진':<12}{'dt':>10}{'최대력':>12}{'충격량':>13}")
    for name, fn in ENGINES:
        rows = []
        for dt in DTS:
            pk, f = fn(12.0, dt)
            rows.append((dt, pk, float(f.sum() * dt)))
            print(f"{name:<12}{dt:10.5f}{pk:10.1f} N{rows[-1][2]:11.4f} N·s")
        a, b = rows[0], rows[-1]
        dpk = (b[1] / a[1] - 1) * 100
        dim = (b[2] / a[2] - 1) * 100
        print(f"{'':<12}{'변화율':>10}{dpk:+9.0f}%{dim:+11.0f}%"
              f"   최대력 {'수렴' if abs(dpk) < 10 else '발산'}")
        print()

    # --- 검사 2: 퍼짐이 줄어드는가 ---
    print("-" * 76)
    print("검사 2. 정합 전후 엔진 간 퍼짐 (dt = 0.2 ms)")
    print("-" * 76)
    print(f"\n{'v0':>5}  {'최대력 [N] 정합후':^30}  {'퍼짐':^13}   {'충격량 퍼짐':^13}")
    print(f"{'':>5}  {'MuJoCo':>9}{'PyBullet':>10}{'Drake':>9}"
          f"  {'전':>6}{'후':>6}   {'전':>6}{'후':>6}")
    print("-" * 76)
    sp_pk, sp_im = [], []
    verdicts = 0
    for v0 in V0S:
        pks, ims = [], []
        for _, fn in ENGINES:
            pk, f = fn(v0, 0.0002)
            pks.append(pk)
            ims.append(float(f.sum() * 0.0002))
        a, b = spread(BEFORE_PK[v0]), spread(pks)
        c, d = spread(BEFORE_IM[v0]), spread(ims)
        sp_pk.append(b)
        sp_im.append(d)
        vs = [x <= LIMIT for x in pks]
        verdicts += len(set(vs)) > 1
        print(f"{v0:5.1f}  {pks[0]:9.1f}{pks[1]:10.1f}{pks[2]:9.1f}"
              f"  {a:5.1f}x{b:5.1f}x   {c:5.2f}x{d:5.2f}x")
    print("-" * 76)
    print(f"\n정합 후 최대력 퍼짐: {min(sp_pk):.1f} ~ {max(sp_pk):.1f}배"
          f"   (정합 전 7.4 ~ 13.0배)")
    print(f"정합 후 충격량 퍼짐: {min(sp_im):.2f} ~ {max(sp_im):.2f}배"
          f"   (정합 전 1.08 ~ 1.17배)")
    print(f"\n안전 판정이 갈린 조건: {verdicts}/{len(V0S)}  (정합 전 2/5)")


if __name__ == "__main__":
    main()
