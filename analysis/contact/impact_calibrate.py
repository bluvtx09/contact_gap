"""정합 기준을 바꿔 다시 맞춘다 — 정적 관측량은 충돌에 옮겨지지 않는다.

정적 정합은 관통 0.08 mm에서 잰 강성을 관통 35 mm 영역에 갖다 쓴다.
400배 밖의 외삽이고, 실제로 팔이 패드를 뚫고 지나갔다.

여기서는 충돌 영역의 관측량(기준 충돌에서의 최대 관통깊이)으로
다시 정합하고, 세 가지 정합 방식을 나란히 비교한다.
"""

import numpy as np

REF_V0 = 9.0            # 기준 충돌 속도 [rad/s]
TARGET_PEN = 0.5e-3     # 기준 충돌에서의 목표 최대 관통깊이 [m]
T_END = 0.30
HOLD_TAU = 0.30
Q_START = 0.1250
MJ_SOLIMP = "0.9 0.95 0.001"
V0S = [3.0, 6.0, 9.0, 12.0, 15.0]
LIMIT = 140.0


# ---------- 충돌 시 최대 관통깊이 ----------------------------------

def pen_mujoco(k, v0=REF_V0, dt=0.0002):
    import contact_model as CM
    import mujoco
    CM.BASE["dt"] = dt
    sr = f"-{k:.6g} -{2 * np.sqrt(k * 0.45):.6g}"
    m = CM.build(pad_solref=sr, tip_solref=sr,
                 pad_solimp=MJ_SOLIMP, tip_solimp=MJ_SOLIMP)
    d = mujoco.MjData(m)
    d.qpos[0] = Q_START
    d.qvel[0] = v0
    pen, fmax, fs = 0.0, 0.0, []
    for _ in range(int(T_END / dt)):
        d.ctrl[0] = HOLD_TAU
        mujoco.mj_step(m, d)
        f = 0.0
        for c in range(d.ncon):
            ft = np.zeros(6)
            mujoco.mj_contactForce(m, d, c, ft)
            f += abs(ft[0])
            pen = max(pen, -d.contact[c].dist)
        fs.append(f)
        fmax = max(fmax, f)
    return pen, fmax, np.array(fs), float(d.qpos[0])


def pen_pybullet(k, v0=REF_V0, dt=0.0002):
    import pybullet as p
    import cross_engine as X
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
    for body, link in [(arm, 0), (pad, -1)]:
        p.changeDynamics(body, link, lateralFriction=X.MU,
                         contactStiffness=k,
                         contactDamping=2 * np.sqrt(k * 0.45),
                         physicsClientId=cid)
    p.resetJointState(arm, 0, Q_START, v0, physicsClientId=cid)
    p.setJointMotorControl2(arm, 0, p.VELOCITY_CONTROL, force=X.J_FRIC,
                            physicsClientId=cid)
    pen, fmax, fs = 0.0, 0.0, []
    for _ in range(int(T_END / dt)):
        p.setJointMotorControl2(arm, 0, p.TORQUE_CONTROL, force=HOLD_TAU,
                                physicsClientId=cid)
        p.stepSimulation(physicsClientId=cid)
        cps = p.getContactPoints(arm, pad, physicsClientId=cid)
        f = sum(abs(c[9]) for c in cps)
        pen = max([pen] + [-c[8] for c in cps])
        fs.append(f)
        fmax = max(fmax, f)
    q = p.getJointState(arm, 0, physicsClientId=cid)[0]
    p.disconnect(physicsClientId=cid)
    return pen, fmax, np.array(fs), float(q)


def pen_drake(k, v0=REF_V0, dt=0.0002):
    from pydrake.systems.framework import DiagramBuilder
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.multibody.parsing import Parser
    from pydrake.systems.analysis import Simulator
    import drake_engine as DK

    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=dt)
    Parser(plant).AddModelsFromString(DK.SDF.format(stiff=k, diss=10.0), "sdf")
    plant.Finalize()
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    pc = plant.GetMyMutableContextFromRoot(ctx)
    j = plant.GetJointByName("j")
    j.set_angle(pc, Q_START)
    j.set_angular_rate(pc, v0)
    plant.get_actuation_input_port().FixValue(pc, np.array([HOLD_TAU]))
    sim = Simulator(diagram, ctx)
    sim.Initialize()
    port = plant.get_contact_results_output_port()
    pen, fmax, fs = 0.0, 0.0, []
    n = int(T_END / dt)
    for step in range(1, n + 1):
        sim.AdvanceTo(step * dt)
        cr = port.Eval(plant.GetMyContextFromRoot(sim.get_context()))
        f = 0.0
        for i in range(cr.num_point_pair_contacts()):
            info = cr.point_pair_contact_info(i)
            f += float(np.linalg.norm(info.contact_force()))
            pen = max(pen, float(info.point_pair().depth))
        fs.append(f)
        fmax = max(fmax, f)
    q = float(j.get_angle(plant.GetMyContextFromRoot(sim.get_context())))
    return pen, fmax, np.array(fs), q


PROBES = [("MuJoCo", pen_mujoco), ("PyBullet", pen_pybullet),
          ("Drake", pen_drake)]


def calibrate_impact(fn, lo, hi, iters=13):
    """기준 충돌에서 최대 관통이 목표가 되는 파라미터를 찾는다."""
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        pen, _, _, _ = fn(mid)
        if pen > TARGET_PEN:
            lo = mid       # 너무 무름 → 강성 올림
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def main():
    print("=" * 76)
    print("충돌 영역 정합 — 기준 충돌의 최대 관통깊이로 맞춘다")
    print("=" * 76)
    print(f"\n기준 충돌 v0 = {REF_V0} rad/s, 목표 최대 관통 {TARGET_PEN*1e3:.2f} mm\n")
    print(f"{'엔진':<11}{'정합 파라미터':>16}{'실제 관통':>12}{'최대력':>11}{'최종각':>10}")
    print("-" * 76)
    cal = {}
    for name, fn in PROBES:
        k = calibrate_impact(fn, 1e2, 1e9)
        pen, fmax, _, q = fn(k)
        cal[name] = k
        thru = " 관통이탈" if q > 0.30 else ""
        print(f"{name:<11}{k:16.4g}{pen*1e3:10.3f} mm{fmax:9.1f} N{q:9.4f}{thru}")
    print("-" * 76)

    print(f"\n{'-' * 76}")
    print("정합 방식 3종 비교 — 엔진 간 최대력 퍼짐")
    print("-" * 76)
    print(f"\n{'v0':>5}  {'충돌정합 최대력 [N]':^30}   {'퍼짐 비교':^26}")
    print(f"{'':>5}  {'MuJoCo':>9}{'PyBullet':>10}{'Drake':>9}"
          f"   {'정합없음':>9}{'정적정합':>9}{'충돌정합':>9}")
    print("-" * 76)
    BEFORE = {3.0: 9.6, 6.0: 12.2, 9.0: 10.0, 12.0: 7.4, 15.0: 13.0}
    STATIC = {3.0: 40.5, 6.0: 49.7, 9.0: 53.7, 12.0: 1.9, 15.0: 1.4}
    sp, dis = [], 0
    for v0 in V0S:
        pks = []
        for name, fn in PROBES:
            _, fmax, _, _ = fn(cal[name], v0=v0)
            pks.append(fmax)
        s = max(pks) / min(pks)
        sp.append(s)
        vs = [x <= LIMIT for x in pks]
        dis += len(set(vs)) > 1
        print(f"{v0:5.1f}  {pks[0]:9.1f}{pks[1]:10.1f}{pks[2]:9.1f}"
              f"   {BEFORE[v0]:8.1f}x{STATIC[v0]:8.1f}x{s:8.1f}x")
    print("-" * 76)
    print(f"\n충돌정합 후 퍼짐: {min(sp):.1f} ~ {max(sp):.1f}배")
    print(f"  (정합없음 7.4~13.0배, 정적정합 1.4~53.7배)")
    print(f"\n안전 판정이 갈린 조건: {dis}/{len(V0S)}")


if __name__ == "__main__":
    main()
