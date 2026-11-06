# mujoco를 설치한 후, 실행해보는 코드입니다.
import time

import mujoco
import mujoco.viewer


# MJCF는 MuJoCo의 시뮬레이션 장면을 정의하는 XML 형식입니다.
SCENE_XML = """
<mujoco model="pick_agent_basic_test">

    <!-- 시간 간격은 0.002초, 중력은 아래 방향으로 설정합니다. -->
    <option timestep="0.002" gravity="0 0 -9.81"/>

    <worldbody>

        <!-- 장면을 비추는 조명입니다. -->
        <light pos="0 0 3" dir="0 0 -1"/>

        <!-- 바닥입니다. -->
        <geom
            name="floor"
            type="plane"
            size="2 2 0.1"
            rgba="0.85 0.85 0.85 1"
        />

        <!-- 바닥 위 0.5m 높이에서 시작하는 큐브입니다. -->
        <body name="blue_cube" pos="0 0 0.5">

            <!-- 위치와 회전이 자유로워 중력에 따라 떨어집니다. -->
            <freejoint/>

            <!-- box의 size는 각 축 길이의 절반입니다.
                 따라서 실제 큐브 크기는 6cm × 6cm × 6cm입니다. -->
            <geom
                name="cube"
                type="box"
                size="0.03 0.03 0.03"
                mass="0.1"
                rgba="0.1 0.3 0.9 1"
            />

        </body>

    </worldbody>
</mujoco>
"""


def main():
    # 모델은 물체 구조, 질량, 중력 등의 설정을 담습니다.
    model = mujoco.MjModel.from_xml_string(SCENE_XML)

    # 데이터는 현재 위치, 속도, 시뮬레이션 시간 등을 담습니다.
    data = mujoco.MjData(model)

    # 초기 위치를 화면에 표시할 수 있도록 계산합니다.
    mujoco.mj_forward(model, data)

    print("MuJoCo 시뮬레이션을 시작합니다.")
    print("파란 큐브가 바닥으로 떨어지면 정상입니다.")
    print("뷰어 창을 닫으면 종료됩니다.")

    # Python 코드에서 물리 계산을 진행하는 방식의 뷰어입니다.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 큐브 주변이 잘 보이도록 카메라를 설정합니다.
        viewer.cam.lookat[:] = [0, 0, 0.15]
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

        viewer.sync()

        # 시작 위치를 잠깐 보여준 뒤 물리 계산을 시작합니다.
        time.sleep(1)

        while viewer.is_running():
            step_start = time.perf_counter()

            # 물리 시뮬레이션을 한 단계 진행합니다.
            mujoco.mj_step(model, data)

            # 계산 결과를 화면에 반영합니다.
            viewer.sync()

            # 시뮬레이션이 실제 시간과 비슷한 속도로 진행되게 합니다.
            remaining = model.opt.timestep - (
                time.perf_counter() - step_start
            )

            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()