# 이 코드는 MuJoCo Menagerie에서 제공하는 SO-101 모델을 불러와서 
# 시뮬레이션하고, 관절 정보를 출력하며, 뷰어를 통해 모델을 시각화합니다.

import time

import mujoco
import mujoco.viewer
import mujoco_menagerie as menagerie


def main():
    print("SO-101 모델을 불러오는 중입니다.")
    print("최초 실행 시 모델 다운로드를 위해 인터넷 연결이 필요합니다.")

    # SO-101 모델과 바닥·조명이 포함된 기본 장면을 불러옵니다.
    # 최초 다운로드 후에는 사용자 캐시에 저장된 모델을 사용합니다.
    model = menagerie.load("robotstudio_so101")

    # 관절 위치, 속도, 제어 입력 등 실행 상태를 생성합니다.
    data = mujoco.MjData(model)

    # 모델에 초기 자세가 정의되어 있으면 첫 번째 자세를 적용합니다.
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    # 초기 상태에서 링크 위치 등을 계산합니다.
    mujoco.mj_forward(model, data)

    print("\n모델 로딩 완료")
    print(f"관절 개수: {model.njnt}")
    print(f"액추에이터 개수: {model.nu}")

    # 모델에 정의된 관절 이름을 확인합니다.
    print("\n관절 목록:")

    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        print(f"  {joint_id}: {name}")

    print("\n뷰어 창을 닫으면 종료됩니다.")

    # 실제 로봇이 아닌 MuJoCo 속 모델을 표시합니다.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 로봇 전체를 볼 수 있도록 카메라를 조정합니다.
        viewer.cam.lookat[:] = model.stat.center
        viewer.cam.distance = max(model.stat.extent * 1.5, 0.6)
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

        while viewer.is_running():
            step_start = time.perf_counter()

            # 물리 시뮬레이션을 한 단계 진행합니다.
            # 아직 관절 목표값을 변경하는 동작 코드는 없습니다.
            mujoco.mj_step(model, data)

            # 계산된 상태를 화면에 표시합니다.
            viewer.sync()

            # 실제 시간에 가까운 속도로 진행합니다.
            remaining = model.opt.timestep - (
                time.perf_counter() - step_start
            )

            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()