import numpy as np
import mujoco

XML = """
<mujoco model="contact_gap">
  <option timestep="{dt}" integrator="RK4" cone="{cone}"/>
  <default>
    <geom condim="{condim}"/>
  </default>
  <worldbody>
    <body name="link" pos="0 0 0.5">
      <joint name="j" type="hinge" axis="0 0 1" damping="{jdamp}" frictionloss="{jfric}"/>
      <geom name="arm" type="capsule" fromto="0 0 0 0.30 0 0" size="0.015"
            mass="{mass}" solref="{tip_solref}" solimp="{tip_solimp}" friction="{mu} 0.005 0.0001"/>
    </body>
    <body name="pad" pos="0.285 0.075 0.5">
      <geom name="pad" type="box" size="0.06 0.02 0.05"
            solref="{pad_solref}" solimp="{pad_solimp}" friction="{mu} 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j" gear="1" ctrlrange="-20 20"/>
  </actuator>
  <sensor>
    <jointpos joint="j"/>
    <jointvel joint="j"/>
  </sensor>
</mujoco>
"""

BASE = dict(
    dt=0.0002,
    cone="pyramidal",
    condim=3,
    jdamp=0.02,
    jfric=0.010,
    mass=0.45,
    mu=0.55,
    tip_solref="0.006 1.0",
    tip_solimp="0.95 0.99 0.001",
    pad_solref="0.020 1.0",
    pad_solimp="0.95 0.99 0.001",
)


def build(**over):
    p = dict(BASE)
    p.update(over)
    return mujoco.MjModel.from_xml_string(XML.format(**p))


def rollout(model, tau_cmd, torque_scale=1.0, dt_out=0.001, q0=0.0, v0=0.0):
    """tau_cmd: callable(t) -> commanded torque. torque_scale: actuator loss factor."""
    data = mujoco.MjData(model)
    data.qpos[0] = q0
    data.qvel[0] = v0
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    n_sub = max(1, int(round(dt_out / dt)))
    n_out = int(round(1.0 / dt_out))

    t_log, q_log, v_log, f_log, tau_log = [], [], [], [], []
    for k in range(n_out):
        t = k * dt_out
        u = tau_cmd(t) * torque_scale
        for _ in range(n_sub):
            data.ctrl[0] = u
            mujoco.mj_step(model, data)
        fn = 0.0
        for c in range(data.ncon):
            ft = np.zeros(6)
            mujoco.mj_contactForce(model, data, c, ft)
            fn += abs(ft[0])
        t_log.append(t)
        q_log.append(data.qpos[0])
        v_log.append(data.qvel[0])
        f_log.append(fn)
        tau_log.append(u)

    return dict(
        t=np.array(t_log),
        q=np.array(q_log),
        v=np.array(v_log),
        f=np.array(f_log),
        tau=np.array(tau_log),
    )


def excitation(kind):
    """Approach-velocity conditions: how the arm is driven into the pad."""
    if kind == "soft_press":
        return lambda t: 0.35 * min(1.0, t / 0.25)
    if kind == "medium_tap":
        return lambda t: 1.1 if t < 0.30 else 0.35
    if kind == "fast_strike":
        return lambda t: 2.6 if t < 0.18 else 0.5
    raise ValueError(kind)


EXCITATIONS = ["soft_press", "medium_tap", "fast_strike"]
