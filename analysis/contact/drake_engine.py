"""세 번째 엔진 — Drake.

MuJoCo(연성 접촉)와 PyBullet(강체 충격량)에 이어,
접촉 모델 계열이 또 다른 Drake에서 같은 충돌을 돌린다.

질문: 최대 접촉력의 시간간격 발산이 PyBullet 고유 성질인가,
아니면 접촉을 강체로 푸는 방식 전반의 성질인가?
"""

import numpy as np

L = 0.30
R = 0.015
MASS = 0.45
J_DAMP = 0.02
MU = 0.55
PAD_POS = (0.285, 0.075, 0.0)
PAD_HALF = (0.06, 0.02, 0.05)
Q_START = 0.1250
HOLD_TAU = 0.30
T_END = 0.30

SDF = f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="rig">
    <link name="link">
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <pose>{L/2} 0 0 0 0 0</pose>
        <mass>{MASS}</mass>
        <inertia>
          <ixx>{MASS*R*R/2:.8f}</ixx><iyy>{MASS*L*L/12:.8f}</iyy>
          <izz>{MASS*L*L/12:.8f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="arm_col">
        <pose>{L/2} 0 0 0 1.5707963 0</pose>
        <geometry><cylinder><radius>{R}</radius><length>{L}</length></cylinder></geometry>
        <drake:proximity_properties>
          <drake:point_contact_stiffness>{{stiff}}</drake:point_contact_stiffness>
          <drake:hunt_crossley_dissipation>{{diss}}</drake:hunt_crossley_dissipation>
          <drake:mu_dynamic>{MU}</drake:mu_dynamic>
          <drake:mu_static>{MU}</drake:mu_static>
        </drake:proximity_properties>
      </collision>
    </link>
    <link name="pad">
      <pose>{PAD_POS[0]} {PAD_POS[1]} {PAD_POS[2]} 0 0 0</pose>
      <inertial><mass>1</mass>
        <inertia><ixx>1</ixx><iyy>1</iyy><izz>1</izz>
        <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      <collision name="pad_col">
        <geometry><box><size>{PAD_HALF[0]*2} {PAD_HALF[1]*2} {PAD_HALF[2]*2}</size></box></geometry>
        <drake:proximity_properties>
          <drake:point_contact_stiffness>{{stiff}}</drake:point_contact_stiffness>
          <drake:hunt_crossley_dissipation>{{diss}}</drake:hunt_crossley_dissipation>
          <drake:mu_dynamic>{MU}</drake:mu_dynamic>
          <drake:mu_static>{MU}</drake:mu_static>
        </drake:proximity_properties>
      </collision>
    </link>
    <joint name="j" type="revolute">
      <parent>world</parent><child>link</child>
      <axis><xyz>0 0 1</xyz>
        <limit><lower>-10</lower><upper>10</upper></limit>
        <dynamics><damping>{J_DAMP}</damping></dynamics>
      </axis>
    </joint>
    <joint name="pad_fix" type="fixed">
      <parent>world</parent><child>pad</child>
    </joint>
  </model>
</sdf>
"""


def run_drake(v0, dt, stiff=1e6, diss=10.0):
    from pydrake.systems.framework import DiagramBuilder
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.multibody.parsing import Parser
    from pydrake.systems.analysis import Simulator

    builder = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(builder, time_step=dt)
    Parser(plant).AddModelsFromString(
        SDF.format(stiff=stiff, diss=diss), "sdf")
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
    fs = []
    n = int(T_END / dt)
    for k in range(1, n + 1):
        sim.AdvanceTo(k * dt)
        cr = port.Eval(plant.GetMyContextFromRoot(sim.get_context()))
        f = 0.0
        for i in range(cr.num_point_pair_contacts()):
            f += float(np.linalg.norm(
                cr.point_pair_contact_info(i).contact_force()))
        fs.append(f)
    fs = np.array(fs)
    return float(fs.max()), fs


if __name__ == "__main__":
    for dt in [0.0008, 0.0002]:
        pk, f = run_drake(12.0, dt)
        print(f"dt={dt:.4f}  peak={pk:8.1f} N  impulse={f.sum()*dt:.4f} N·s"
              f"  contact_steps={(f > 1).sum()}")
