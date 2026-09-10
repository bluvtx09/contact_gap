"""엔진 교차 검증 — 같은 물리 시나리오를 두 엔진에서 돌린다.

질문: 널리 쓰이는 두 엔진이 각자의 기본 접촉 설정으로
같은 충돌의 최대 접촉력을 얼마나 다르게 예측하는가?

여기에는 '정답'이 없다. 두 엔진이 크게 다르다면,
어느 쪽 숫자로 안전 판정을 내려도 근거가 없다는 뜻이다.
"""

import os
import sys
import contextlib
import numpy as np


@contextlib.contextmanager
def quiet():
    """PyBullet의 C 레벨 경고 출력을 막는다."""
    fd = sys.stderr.fileno()
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(1)
    os.dup2(devnull, fd)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(saved, fd)
        os.dup2(old, 1)
        os.close(devnull)
        os.close(saved)
        os.close(old)

# --- 공통 물리 사양 -------------------------------------------------
L = 0.30          # 링크 길이 [m]
R = 0.015         # 링크 반경 [m]
MASS = 0.45       # 링크 질량 [kg]
J_DAMP = 0.02     # 관절 점성 [N·m·s/rad]
J_FRIC = 0.010    # 관절 건마찰 [N·m]
MU = 0.55
PAD_POS = (0.285, 0.075, 0.0)
PAD_HALF = (0.06, 0.02, 0.05)
Q_START = 0.1250
HOLD_TAU = 0.30
DT = 0.0002
T_END = 0.30

URDF = f"""<?xml version="1.0"?>
<robot name="arm">
  <link name="base">
    <inertial><mass value="0"/><inertia ixx="0" iyy="0" izz="0" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="link">
    <inertial>
      <origin xyz="{L/2} 0 0"/>
      <mass value="{MASS}"/>
      <inertia ixx="{MASS*R*R/2:.8f}" iyy="{MASS*L*L/12:.8f}"
               izz="{MASS*L*L/12:.8f}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision>
      <origin xyz="{L/2} 0 0" rpy="0 1.5707963 0"/>
      <geometry><cylinder radius="{R}" length="{L}"/></geometry>
    </collision>
  </link>
  <joint name="j" type="continuous">
    <parent link="base"/><child link="link"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
  </joint>
</robot>
"""

PAD_URDF = f"""<?xml version="1.0"?>
<robot name="pad">
  <link name="pad">
    <inertial><mass value="0"/><inertia ixx="0" iyy="0" izz="0" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision>
      <geometry><box size="{PAD_HALF[0]*2} {PAD_HALF[1]*2} {PAD_HALF[2]*2}"/></geometry>
    </collision>
  </link>
</robot>
"""


def run_pybullet(v0, contact_stiffness=None, contact_damping=None, dt=None):
    import pybullet as p
    dt = dt or DT
    with quiet():
        cid = p.connect(p.DIRECT)
    p.resetSimulation(physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.setTimeStep(dt, physicsClientId=cid)

    d = "/tmp/xengine"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/arm.urdf", "w").write(URDF)
    open(f"{d}/pad.urdf", "w").write(PAD_URDF)

    arm = p.loadURDF(f"{d}/arm.urdf", [0, 0, 0], useFixedBase=True,
                     physicsClientId=cid)
    pad = p.loadURDF(f"{d}/pad.urdf", PAD_POS, useFixedBase=True,
                     physicsClientId=cid)

    kw = dict(lateralFriction=MU, jointDamping=J_DAMP)
    if contact_stiffness is not None:
        kw.update(contactStiffness=contact_stiffness,
                  contactDamping=contact_damping)
    p.changeDynamics(arm, 0, physicsClientId=cid, **kw)
    p.changeDynamics(pad, -1, lateralFriction=MU, physicsClientId=cid,
                     **({} if contact_stiffness is None else
                        dict(contactStiffness=contact_stiffness,
                             contactDamping=contact_damping)))

    p.resetJointState(arm, 0, Q_START, v0, physicsClientId=cid)
    p.setJointMotorControl2(arm, 0, p.VELOCITY_CONTROL, force=J_FRIC,
                            physicsClientId=cid)

    peak, fs = 0.0, []
    n = int(T_END / dt)
    for _ in range(n):
        p.setJointMotorControl2(arm, 0, p.TORQUE_CONTROL, force=HOLD_TAU,
                                physicsClientId=cid)
        p.stepSimulation(physicsClientId=cid)
        cps = p.getContactPoints(arm, pad, physicsClientId=cid)
        f = sum(abs(c[9]) for c in cps)
        fs.append(f)
        peak = max(peak, f)
    p.disconnect(physicsClientId=cid)
    return peak, np.array(fs)


def run_mujoco(v0, solref, dt=None):
    import contact_model as CM
    dt = dt or DT
    CM.BASE["dt"] = dt
    m = CM.build(pad_solref=solref, tip_solref=solref, mu=MU,
                 jfric=J_FRIC, jdamp=J_DAMP, mass=MASS)
    r = CM.rollout(m, lambda t: HOLD_TAU, torque_scale=1.0,
                   dt_out=dt, q0=Q_START, v0=v0)
    n = int(T_END / dt)
    return float(r["f"][:n].max()), r["f"][:n]


def impulse(f, dt):
    """접촉력의 시간적분 [N·s]."""
    return float(np.sum(f) * dt)


def _drake(v, dt, stiff):
    from drake_engine import run_drake
    return run_drake(v, dt, stiff=stiff, diss=10.0)


ENGINES = [
    ("MuJoCo", lambda v, dt: run_mujoco(v, "0.006 1.0", dt)),
    ("PyBullet 기본", lambda v, dt: run_pybullet(v, dt=dt)),
    ("PyBullet 강성지정", lambda v, dt: run_pybullet(v, 3e4, 3e2, dt=dt)),
    ("Drake k=1e5", lambda v, dt: _drake(v, dt, 1e5)),
    ("Drake k=1e6", lambda v, dt: _drake(v, dt, 1e6)),
]

# 비교용 대표 설정 (엔진당 하나)
MAIN = [ENGINES[0], ENGINES[1], ENGINES[3]]

DTS = [0.0004, 0.0002, 0.0001, 0.00005]
V0S = [3.0, 6.0, 9.0, 12.0, 15.0]
LIMIT = 140.0


def convergence():
    print("=" * 74)
    print("검사 1. 시간간격 수렴성 (v0 = 12 rad/s)")
    print("=" * 74)
    print("\n간격을 촘촘히 했을 때 값이 수렴하지 않으면,")
    print("그 숫자는 물리 예측이 아니라 적분 설정의 부산물이다.\n")
    print(f"{'엔진':<19}{'dt':>10}{'최대력':>12}{'충격량':>13}")
    print("-" * 74)
    verdict = {}
    for name, fn in ENGINES:
        rows = []
        for dt in DTS:
            pk, f = fn(12.0, dt)
            rows.append((dt, pk, impulse(f, dt)))
            print(f"{name:<19}{dt:10.5f}{pk:10.1f} N{rows[-1][2]:11.4f} N·s")
        a, b = rows[0], rows[-1]
        dpk = (b[1] / a[1] - 1) * 100
        dim = (b[2] / a[2] - 1) * 100
        verdict[name] = (dpk, dim)
        print(f"{'':<19}{'변화율':>10}{dpk:+9.0f}%{dim:+11.0f}%")
        print()
    print("-" * 74)
    print("\n판정 (변화율 10% 이내를 수렴으로 봄):")
    for name, (dpk, dim) in verdict.items():
        print(f"  {name:<19} 최대력 {'수렴' if abs(dpk) < 10 else '발산':<4}"
              f"   충격량 {'수렴' if abs(dim) < 10 else '발산'}")


def spread():
    print(f"\n{'=' * 74}")
    print("검사 2. 엔진 간 일치도 (dt = 0.2 ms 고정)")
    print("=" * 74)
    print(f"\n{'v0':>5}  {'최대력 [N]':^30}   {'충격량 [N·s]':^28}")
    print(f"{'':>5}  {'MuJoCo':>8}{'PyBullet':>10}{'Drake':>8}{'퍼짐':>5}"
          f"   {'MuJoCo':>8}{'PyBullet':>9}{'Drake':>7}{'퍼짐':>6}")
    print("-" * 74)
    sp_pk, sp_im = [], []
    for v0 in V0S:
        pks, ims = [], []
        for _, fn in MAIN:
            pk, f = fn(v0, 0.0002)
            pks.append(pk)
            ims.append(impulse(f, 0.0002))
        a = max(pks) / min(pks)
        b = max(ims) / min(ims)
        sp_pk.append(a)
        sp_im.append(b)
        print(f"{v0:5.1f}  {pks[0]:8.1f}{pks[1]:10.1f}{pks[2]:8.1f}{a:4.1f}x"
              f"   {ims[0]:8.3f}{ims[1]:9.3f}{ims[2]:7.3f}{b:5.2f}x")
    print("-" * 74)
    print(f"\n최대력 엔진간 퍼짐: {min(sp_pk):.1f} ~ {max(sp_pk):.1f}배")
    print(f"충격량 엔진간 퍼짐: {min(sp_im):.2f} ~ {max(sp_im):.2f}배"
          f"  (최대 {(max(sp_im) - 1) * 100:.0f}% 차이)")
    print(f"\n재현성 차이: 약 {max(sp_pk) / max(sp_im):.0f}배")


def verdicts():
    print(f"\n{'=' * 74}")
    print(f"검사 3. 안전 판정 일치 여부 (문턱 {LIMIT:.0f} N, dt = 0.2 ms)")
    print("=" * 74)
    print(f"\n{'v0':>5}{'MuJoCo':>10}{'PyBullet':>10}{'Drake':>10}  일치?")
    print("-" * 50)
    n_dis = 0
    for v0 in V0S:
        vs = [fn(v0, 0.0002)[0] <= LIMIT for _, fn in MAIN]
        lab = ["안전" if x else "초과" for x in vs]
        ok = len(set(vs)) == 1
        n_dis += (not ok)
        print(f"{v0:5.1f}{lab[0]:>10}{lab[1]:>10}{lab[2]:>10}"
              f"  {'예' if ok else '*** 아니오 ***'}")
    print("-" * 50)
    print(f"\n판정이 갈린 조건: {n_dis}/{len(V0S)}")


def main():
    print("=" * 74)
    print("엔진 교차 검증 — 최대 접촉력은 엔진 간에 옮겨지지 않는다")
    print("=" * 74)
    print(f"\n링크 {L} m / {MASS} kg, 마찰 {MU}")
    print("\n어느 엔진이 옳은지 판정하지 않는다.")
    print("엔진들이 크게 다르다면, 어느 쪽 숫자로도")
    print("안전 판정의 근거를 삼을 수 없다는 것이 요점이다.\n")
    convergence()
    spread()
    verdicts()


if __name__ == "__main__":
    main()
