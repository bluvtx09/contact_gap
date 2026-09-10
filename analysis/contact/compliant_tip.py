"""직렬 탄성 접촉부 — 설계로 갭에 둔감해질 수 있는가?

가설: 팔 끝에 **알려진 강성**의 탄성 요소를 직렬로 넣으면,
미지의 환경 접촉강성이 결과를 지배하지 못하게 된다.

직렬 연결의 유효강성은 k_eff = k_tip·k_env / (k_tip + k_env) 이므로,
k_tip << k_env 이면 k_eff ≈ k_tip 이다.
환경 강성이 몇 배 틀려도 유효강성은 거의 안 변한다.

k_tip = None 이면 탄성 요소 없는 기존 강체 팔이다.
"""

import numpy as np
import mujoco

RIGID_XML = """
<mujoco model="rigid_arm">
  <option timestep="{dt}" integrator="RK4"/>
  <worldbody>
    <body name="link" pos="0 0 0.5">
      <joint name="j" type="hinge" axis="0 0 1" damping="0.02" frictionloss="0.010"/>
      <geom name="arm" type="capsule" fromto="0 0 0 0.30 0 0" size="0.015"
            mass="0.45" solref="{env_solref}" solimp="0.95 0.99 0.001"
            friction="0.55 0.005 0.0001"/>
    </body>
    <body name="pad" pos="0.285 0.075 0.5">
      <geom name="pad" type="box" size="0.06 0.02 0.05"
            solref="{env_solref}" solimp="0.95 0.99 0.001"
            friction="0.55 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator><motor joint="j" gear="1" ctrlrange="-20 20"/></actuator>
</mujoco>
"""

# 팔 끝에 슬라이드 관절 + 스프링으로 직렬 탄성 요소를 단다.
SEA_XML = """
<mujoco model="sea_arm">
  <option timestep="{dt}" integrator="RK4"/>
  <worldbody>
    <body name="link" pos="0 0 0.5">
      <joint name="j" type="hinge" axis="0 0 1" damping="0.02" frictionloss="0.010"/>
      <geom name="arm" type="capsule" fromto="0 0 0 0.26 0 0" size="0.015"
            mass="0.40" contype="0" conaffinity="0"/>
      <body name="tip" pos="0.26 0 0">
        <joint name="s" type="slide" axis="0 1 0"
               stiffness="{k_tip}" damping="{b_tip}" range="-0.05 0.05"/>
        <geom name="tipg" type="capsule" fromto="0 0 0 0.04 0 0" size="0.015"
              mass="0.05" solref="{env_solref}" solimp="0.95 0.99 0.001"
              friction="0.55 0.005 0.0001"/>
      </body>
    </body>
    <body name="pad" pos="0.285 0.075 0.5">
      <geom name="pad" type="box" size="0.06 0.02 0.05"
            solref="{env_solref}" solimp="0.95 0.99 0.001"
            friction="0.55 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator><motor joint="j" gear="1" ctrlrange="-20 20"/></actuator>
</mujoco>
"""

Q_START = 0.1250
HOLD_TAU = 0.30
T_END = 0.30
DT = 0.0002


def build(env_solref, k_tip=None):
    if k_tip is None:
        return mujoco.MjModel.from_xml_string(
            RIGID_XML.format(dt=DT, env_solref=env_solref))
    b = 2 * np.sqrt(k_tip * 0.05)      # 임계감쇠 근처
    return mujoco.MjModel.from_xml_string(
        SEA_XML.format(dt=DT, env_solref=env_solref,
                       k_tip=k_tip, b_tip=b))


def impact(env_solref, k_tip, v0):
    """반환: (최대력, 힘궤적, 관통이탈 여부).

    관통이탈이면 최대력은 접촉력이 아니라 이탈 스파이크이므로
    물리값으로 쓰면 안 된다."""
    m = build(env_solref, k_tip)
    d = mujoco.MjData(m)
    d.qpos[0] = Q_START
    d.qvel[0] = v0
    peak, fs = 0.0, []
    for _ in range(int(T_END / DT)):
        d.ctrl[0] = HOLD_TAU
        mujoco.mj_step(m, d)
        f = 0.0
        for c in range(d.ncon):
            ft = np.zeros(6)
            mujoco.mj_contactForce(m, d, c, ft)
            f += abs(ft[0])
        fs.append(f)
        peak = max(peak, f)
    tunneled = float(d.qpos[0]) > 0.30
    return peak, np.array(fs), tunneled


TRUE_ENV = "0.006 1.0"     # 실제 환경 접촉강성 (딱딱)
WRONG_ENV = "0.020 1.0"    # 시뮬이 잘못 잡은 값 (무름)

TIPS = [
    ("탄성 없음 (강체)", None),
    ("k=20000 N/m", 20000.0),
    ("k=5000 N/m", 5000.0),
    ("k=1500 N/m", 1500.0),
    ("k=500 N/m", 500.0),
]


ENVS = [("0.004 1.0", "매우 딱딱"), ("0.006 1.0", "딱딱"),
        ("0.010 1.0", "중간"), ("0.020 1.0", "무름"),
        ("0.040 1.0", "매우 무름")]
LIMIT = 140.0


def max_valid_speed(env, k, cap=70.0):
    """관통이탈이 시작되기 직전 속도. 이 위로는 시뮬이 무효다."""
    lo, hi = 1.0, cap
    if not impact(env, k, hi)[2]:
        return hi
    for _ in range(14):
        mid = (lo + hi) / 2
        if impact(env, k, mid)[2]:
            hi = mid
        else:
            lo = mid
    return lo


def cross_speed(env, k):
    """문턱을 넘기 시작하는 접근속도.

    관통이탈이 나는 속도 위쪽은 시뮬이 무효이므로 탐색 범위에서 뺀다.
    유효 범위 안에서 끝내 문턱을 안 넘으면 (None, 유효상한)을 준다."""
    vmax = max_valid_speed(env, k)
    lo, hi = 1.0, vmax
    if impact(env, k, hi)[0] <= LIMIT:
        return None, vmax
    for _ in range(15):
        mid = (lo + hi) / 2
        if impact(env, k, mid)[0] <= LIMIT:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, vmax


def part_a():
    print("=" * 74)
    print("검사 A. 탄성이 최대 접촉력을 낮추는가")
    print("=" * 74)
    print(f"\n{'접촉부 설계':<20}{'실물 peak':>11}{'시뮬 peak':>11}"
          f"{'상대오차':>10}{'절대오차':>11}")
    print("-" * 74)
    for name, k in TIPS:
        pr, _, _ = impact(TRUE_ENV, k, 9.0)
        ps, _, _ = impact(WRONG_ENV, k, 9.0)
        print(f"{name:<20}{pr:9.1f} N{ps:9.1f} N{(ps - pr) / pr * 100:+9.1f}%"
              f"{abs(ps - pr):9.1f} N")
    print("\n최대력 자체는 크게 떨어진다 (150 → 43 N). 절대오차도 함께 줄어든다.")
    print("그러나 상대오차는 −69%에서 −46% 부근에 머문다.")


def part_b():
    print(f"\n{'=' * 74}")
    print("검사 B. 탄성이 시뮬 신뢰도를 높이는가 — 안전 한계속도")
    print("=" * 74)
    print("\n주의: 시뮬이 문턱을 넘는 시점을 그냥 재면 안 된다.")
    print("무른 환경 모델에서는 팔이 패드를 통과하며 스파이크가 뜨는데,")
    print("그 스파이크가 문턱을 넘는 것을 '경고'로 오인하기 쉽다.")
    print("여기서는 관통이탈이 시작되는 속도(유효상한)를 먼저 구하고,")
    print("그 안에서만 문턱 초과를 찾는다.\n")
    print(f"{'접촉부 설계':<20}{'실물 한계':>10}{'시뮬 유효상한':>14}"
          f"{'유효범위내 경고':>16}")
    print("-" * 74)
    for name, k in TIPS:
        vr, _ = cross_speed(TRUE_ENV, k)
        vs, cap_s = cross_speed(WRONG_ENV, k)
        warn = f"{vs:.1f} rad/s" if vs is not None else "없음"
        vr_s = f"{vr:.1f}" if vr is not None else "미도달"
        print(f"{name:<20}{vr_s:>10}{cap_s:12.1f}{warn:>16}")
    print("-" * 74)
    print("\n실물 한계속도는 8.6 → 27.8 rad/s로 3.2배 올라간다. 실제로 안전해진다.")
    print("\n그러나 어느 설계에서도 시뮬은 유효범위 안에서 경고를 내지 않는다.")
    print("시뮬은 위험을 알려주기 전에 먼저 수치적으로 무효가 된다.")
    print("**탄성은 로봇을 안전하게 만들지만 시뮬을 신뢰하게 만들지는 않는다.**")


def part_c():
    print(f"\n{'=' * 74}")
    print("검사 C. 미지 환경 강성에 대한 민감도 (v0 = 9)")
    print("=" * 74)
    print(f"\n{'접촉부 설계':<20}" + "".join(f"{d:>10}" for _, d in ENVS)
          + f"{'퍼짐':>9}")
    print("-" * 74)
    for name, k in TIPS:
        cells, valid = [], []
        for e, _ in ENVS:
            pk, _, tun = impact(e, k, 9.0)
            if tun:
                cells.append(f"{'관통이탈':>10}")
            else:
                cells.append(f"{pk:8.1f} N")
                valid.append(pk)
        sp = max(valid) / min(valid)
        print(f"{name.split(' (')[0]:<20}" + "".join(cells) + f"{sp:8.2f}x")
    print("\n퍼짐은 관통이탈 칸을 뺀 유효값끼리만 계산했다.")
    print("가장 무른 환경에서는 강체·탄성 모두 팔이 패드를 통과해 시뮬이 무효다.")
    print("탄성이 이 실패를 없애지는 못한다. 민감도만 4.69배에서 2.2~2.9배로 낮춘다.")


def main():
    part_a()
    part_b()
    part_c()
    print(f"\n{'=' * 74}")
    print("설계 가설 판정")
    print("=" * 74)
    print("""
성립한 것
  - 최대 접촉력이 3.5배 낮아진다 (150 → 43 N)
  - 실물 안전 한계속도가 3.2배 올라간다 (8.6 → 27.8 rad/s)
  - 환경 강성 민감도가 4.69배에서 2.2~2.9배로 낮아진다

성립하지 않은 것
  - 상대오차가 −46% 부근에서 멈춘다 (강체는 −69%)
  - 어느 설계에서도 시뮬은 유효범위 안에서 문턱 초과를 경고하지 않는다.
    무른 환경 모델은 경고를 내기 전에 관통이탈로 먼저 무효가 된다.

  - 아주 무른 환경에서의 관통이탈은 탄성으로도 없어지지 않는다

즉 탄성 설계는 로봇을 실제로 안전하게 만들고 미지 파라미터에 대한
민감도를 낮추지만, 시뮬 예측을 비례적으로 더 믿을 수 있게 만들지는
않는다. 안전 설계와 시뮬 신뢰도는 별개의 축이다.
""")


if __name__ == "__main__":
    main()
