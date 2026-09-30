"""엔진 간 파라미터 정합 — 이름이 아니라 관측량으로 맞춘다.

세 엔진의 '접촉강성'은 같은 것을 뜻하지 않는다.
  MuJoCo   solref = (시상수, 감쇠비) 또는 음수로 (강성, 감쇠)
  PyBullet contactStiffness [N/m] + contactDamping [N·s/m]
  Drake    point_contact_stiffness [N/m] + Hunt-Crossley 소산 [s/m]

Hunt-Crossley는 f = k·x·(1 + d·ẋ) 형태라 선형 감쇠와 단위부터 다르다.
따라서 nominal 값을 같게 두는 것은 정합이 아니다.

대신 **정적 유효강성**이라는 하나의 관측량으로 맞춘다.
일정 하중으로 눌렀을 때의 정상상태 관통깊이를 재고,
k_eff = F_정상 / δ_정상 이 목표값이 되도록 각 엔진의 파라미터를 조정한다.
"""

import numpy as np

TARGET_K = 1.0e5          # 목표 정적 유효강성 [N/m]
PRESS_TAU = 2.5           # 정합용 정적 압박 토크 [N·m]
SETTLE_T = 0.60           # 정상상태 도달 대기 [s]
CAL_DT = 0.0002

# MuJoCo는 solref(강성)와 별개로 solimp(임피던스)가 유효강성을 좌우한다.
# 정합에서는 solimp를 고정하고 solref 강성만 조정한다.
MJ_SOLIMP = "0.9 0.95 0.001"


# ---------- 관측량 측정: 정적 유효강성 -------------------------------

def keff_mujoco(param):
    """param: 접촉강성 [N/m]. MuJoCo는 solref 음수로 직접 지정한다."""
    import contact_model as CM
    CM.BASE["dt"] = CAL_DT
    solref = f"-{param:.6g} -{2 * np.sqrt(param * 0.45):.6g}"
    m = CM.build(pad_solref=solref, tip_solref=solref,
                 pad_solimp=MJ_SOLIMP, tip_solimp=MJ_SOLIMP)
    import mujoco
    d = mujoco.MjData(m)
    d.qpos[0] = 0.125
    n = int(SETTLE_T / CAL_DT)
    for _ in range(n):
        d.ctrl[0] = PRESS_TAU
        mujoco.mj_step(m, d)
    f, pen = 0.0, 0.0
    for c in range(d.ncon):
        ft = np.zeros(6)
        mujoco.mj_contactForce(m, d, c, ft)
        f += abs(ft[0])
        pen = max(pen, -d.contact[c].dist)
    return (f / pen if pen > 1e-12 else np.nan), f, pen


def keff_pybullet(param):
    """param: contactStiffness [N/m]."""
    import pybullet as p
    import cross_engine as X
    with X.quiet():
        cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.setTimeStep(CAL_DT, physicsClientId=cid)
    damp = 2 * np.sqrt(param * 0.45)
    arm, pad = X.load_pybullet_scene(p, cid, param, damp)
    p.resetJointState(arm, 0, 0.125, 0.0, physicsClientId=cid)
    p.setJointMotorControl2(arm, 0, p.VELOCITY_CONTROL, force=X.J_FRIC,
                            physicsClientId=cid)
    for _ in range(int(SETTLE_T / CAL_DT)):
        p.setJointMotorControl2(arm, 0, p.TORQUE_CONTROL, force=PRESS_TAU,
                                physicsClientId=cid)
        p.stepSimulation(physicsClientId=cid)
    cps = p.getContactPoints(arm, pad, physicsClientId=cid)
    f = sum(abs(c[9]) for c in cps)
    pen = max([-c[8] for c in cps], default=0.0)
    p.disconnect(physicsClientId=cid)
    return (f / pen if pen > 1e-12 else np.nan), f, pen


def keff_drake(param):
    """param: point_contact_stiffness [N/m]."""
    from pydrake.systems.framework import DiagramBuilder
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.multibody.parsing import Parser
    from pydrake.systems.analysis import Simulator
    import drake_engine as DK

    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=CAL_DT)
    Parser(plant).AddModelsFromString(
        DK.SDF.format(stiff=param, diss=10.0), "sdf")
    plant.Finalize()
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    pc = plant.GetMyMutableContextFromRoot(ctx)
    j = plant.GetJointByName("j")
    j.set_angle(pc, 0.125)
    j.set_angular_rate(pc, 0.0)
    plant.get_actuation_input_port().FixValue(pc, np.array([PRESS_TAU]))
    sim = Simulator(diagram, ctx)
    sim.Initialize()
    sim.AdvanceTo(SETTLE_T)
    cr = plant.get_contact_results_output_port().Eval(
        plant.GetMyContextFromRoot(sim.get_context()))
    f, pen = 0.0, 0.0
    for i in range(cr.num_point_pair_contacts()):
        info = cr.point_pair_contact_info(i)
        f += float(np.linalg.norm(info.contact_force()))
        pen = max(pen, float(info.point_pair().depth))
    return (f / pen if pen > 1e-12 else np.nan), f, pen


# ---------- 정합: 목표 유효강성에 맞는 파라미터 탐색 -------------------

def calibrate(fn, lo, hi, target=TARGET_K, iters=14):
    """유효강성이 목표가 되는 파라미터를 이분 탐색한다."""
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        k, _, _ = fn(mid)
        if not np.isfinite(k):
            lo = mid
            continue
        if k < target:
            lo = mid
        else:
            hi = mid
    p = np.sqrt(lo * hi)
    k, f, pen = fn(p)
    return p, k, f, pen


ENGINE_CAL = [
    ("MuJoCo", keff_mujoco, 1e0, 1e8),
    ("PyBullet", keff_pybullet, 1e3, 1e8),
    ("Drake", keff_drake, 1e3, 1e8),
]


def main():
    print("=" * 72)
    print("엔진 정합 — 정적 유효강성 기준")
    print("=" * 72)
    print(f"\n목표 유효강성 {TARGET_K:.0e} N/m")
    print(f"정합용 정적 압박 토크 {PRESS_TAU} N·m, 정상상태 {SETTLE_T} s\n")
    print(f"{'엔진':<10}{'정합 파라미터':>16}{'유효강성':>14}"
          f"{'정상힘':>10}{'관통깊이':>12}")
    print("-" * 72)
    out = {}
    for name, fn, lo, hi in ENGINE_CAL:
        p, k, f, pen = calibrate(fn, lo, hi)
        out[name] = p
        print(f"{name:<10}{p:16.4g}{k:12.4g} N/m{f:9.2f} N{pen * 1e3:10.4f} mm")
    print("-" * 72)
    print("\n정합 전에는 각 엔진에 '합리적 범위'의 값을 임의로 넣었다.")
    print("정합 후에는 세 엔진이 같은 하중에서 같은 깊이만큼 눌린다.")
    return out


if __name__ == "__main__":
    main()
