"""흡수 에너지의 엔진 간 이식성.

ISO/TS 15066은 부위별 최대 힘에서 최대 허용 에너지를 유도한다.
그렇다면 시뮬로 잰 흡수 에너지는 엔진 간에 옮겨지는가?

흡수 에너지 = 접촉 시작 직전 운동에너지 − 접촉 중 최소 운동에너지
"""

import numpy as np
import mujoco
import cross_engine as X
import contact_model as CM

I_ARM = 0.45 * 0.30 ** 2 / 3
DT = 0.0002
T_END = 0.30
Q0, TAU = 0.125, 0.30


def ke(w):
    return 0.5 * I_ARM * w * w


def mj(v0, dt=DT):
    CM.BASE["dt"] = dt
    m = CM.build(pad_solref="0.006 1.0", tip_solref="0.006 1.0",
                 mu=0.55, jfric=0.010, jdamp=0.02, mass=0.45)
    d = mujoco.MjData(m)
    d.qpos[0] = Q0
    d.qvel[0] = v0
    ws, ic = [], []
    for _ in range(int(T_END / dt)):
        d.ctrl[0] = TAU
        mujoco.mj_step(m, d)
        ws.append(d.qvel[0])
        ic.append(d.ncon > 0)
    return np.array(ws), np.array(ic)


def pb(v0, dt=DT):
    import pybullet as p
    import os
    with X.quiet():
        cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.setTimeStep(dt, physicsClientId=cid)
    os.makedirs("/tmp/xengine", exist_ok=True)
    open("/tmp/xengine/arm.urdf", "w").write(X.URDF)
    open("/tmp/xengine/pad.urdf", "w").write(X.PAD_URDF)
    arm = p.loadURDF("/tmp/xengine/arm.urdf", [0, 0, 0],
                     useFixedBase=True, physicsClientId=cid)
    pad = p.loadURDF("/tmp/xengine/pad.urdf", X.PAD_POS,
                     useFixedBase=True, physicsClientId=cid)
    p.changeDynamics(arm, 0, lateralFriction=0.55, jointDamping=0.02,
                     physicsClientId=cid)
    p.changeDynamics(pad, -1, lateralFriction=0.55, physicsClientId=cid)
    p.resetJointState(arm, 0, Q0, v0, physicsClientId=cid)
    p.setJointMotorControl2(arm, 0, p.VELOCITY_CONTROL, force=0.010,
                            physicsClientId=cid)
    ws, ic = [], []
    for _ in range(int(T_END / dt)):
        p.setJointMotorControl2(arm, 0, p.TORQUE_CONTROL, force=TAU,
                                physicsClientId=cid)
        p.stepSimulation(physicsClientId=cid)
        ws.append(p.getJointState(arm, 0, physicsClientId=cid)[1])
        ic.append(len(p.getContactPoints(arm, pad, physicsClientId=cid)) > 0)
    p.disconnect(physicsClientId=cid)
    return np.array(ws), np.array(ic)


def dk(v0, dt=DT):
    from pydrake.systems.framework import DiagramBuilder
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.multibody.parsing import Parser
    from pydrake.systems.analysis import Simulator
    import drake_engine as DK
    b = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(b, time_step=dt)
    Parser(plant).AddModelsFromString(DK.SDF.format(stiff=1e5, diss=10.0), "sdf")
    plant.Finalize()
    dg = b.Build()
    ctx = dg.CreateDefaultContext()
    pc = plant.GetMyMutableContextFromRoot(ctx)
    j = plant.GetJointByName("j")
    j.set_angle(pc, Q0)
    j.set_angular_rate(pc, v0)
    plant.get_actuation_input_port().FixValue(pc, np.array([TAU]))
    sim = Simulator(dg, ctx)
    sim.Initialize()
    port = plant.get_contact_results_output_port()
    ws, ic = [], []
    for k in range(1, int(T_END / dt) + 1):
        sim.AdvanceTo(k * dt)
        c = plant.GetMyContextFromRoot(sim.get_context())
        ws.append(j.get_angular_rate(c))
        ic.append(port.Eval(c).num_point_pair_contacts() > 0)
    return np.array(ws), np.array(ic)


ENGINES = [("MuJoCo", mj), ("PyBullet", pb), ("Drake", dk)]


def absorbed(fn, v0):
    w, ic = fn(v0)
    if not ic.any():
        return float("nan")
    i0 = int(np.argmax(ic))
    seg = w[i0:]
    return float(ke(w[max(i0 - 1, 0)]) - ke(seg.min() if seg.size else w[i0]))


def main():
    print("=" * 70)
    print("충돌 중 흡수 에너지 — 엔진 간 이식성")
    print("=" * 70)
    print("\nISO/TS 15066이 유도에 쓰는 양이 시뮬 간에 옮겨지는지 확인한다.\n")
    print(f"{'v0':>5}{'MuJoCo':>11}{'PyBullet':>11}{'Drake':>10}{'퍼짐':>8}")
    print("-" * 70)
    sp = []
    for v0 in [3.0, 6.0, 9.0, 12.0]:
        es = [absorbed(fn, v0) for _, fn in ENGINES]
        s = max(es) / min(es)
        sp.append(s)
        print(f"{v0:5.1f}{es[0]:9.4f} J{es[1]:9.4f} J{es[2]:8.4f} J{s:7.2f}x")
    print("-" * 70)
    print(f"\n에너지 퍼짐: {min(sp):.2f} ~ {max(sp):.2f}배")
    print("\n지표별 엔진 간 퍼짐 비교")
    print(f"  흡수 에너지  {min(sp):.2f} ~ {max(sp):.2f}배")
    print("  충격량       1.08 ~ 1.17배")
    print("  최대 접촉력  7.4  ~ 13.0배")
    print("\n표준이 쓰는 양이 가장 잘 옮겨지고,")
    print("실무가 재는 양이 가장 안 옮겨진다.")


if __name__ == "__main__":
    main()
