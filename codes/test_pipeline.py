# 역할: API 없이 영상·좌표 변환·물리 집기·도달 실패·중단 처리를 검증합니다.

import tempfile
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(
        0, str(Path(__file__).resolve().parent.parent)
    )

from codes import config as cfg
from codes.agent import Agent, capture, validate_selection
from codes.robot import OperationCancelled, Robot


class TestViewer:
    """화면 없이 제어 루프를 실행하기 위한 뷰어 대체물입니다."""

    def __init__(self):
        """뷰어의 열림 상태를 준비합니다."""
        self.running = True

    def is_running(self):
        """테스트에서 설정한 뷰어 상태를 반환합니다."""
        return self.running

    def sync(self):
        """화면 갱신은 생략합니다."""


def prepare_robot(layout="A"):
    """실시간 대기만 생략한 테스트용 로봇을 만듭니다."""
    robot = Robot(layout=layout)
    viewer = TestViewer()

    def fast_step(unused):
        """물리 계산과 중단 검사는 유지합니다."""
        robot.check_operation(viewer)
        mujoco.mj_step(robot.model, robot.data)
        viewer.sync()
        robot.check_operation(viewer)

    robot.step = fast_step
    return robot, viewer


def synthetic_selection(image):
    """테스트에서만 파란 영역으로 합성 선택 박스를 만듭니다."""
    rgb = np.asarray(image).astype(float)
    mask = (
        (rgb[:, :, 2] > rgb[:, :, 0] * 1.8)
        & (rgb[:, :, 2] > rgb[:, :, 1] * 1.5)
    )
    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise AssertionError("이미지에서 파란 큐브를 찾지 못했습니다.")

    return {
        "label": "blue cube (synthetic test)",
        "box_2d": [
            round(ys.min() / cfg.IMAGE_HEIGHT * 1000),
            round(xs.min() / cfg.IMAGE_WIDTH * 1000),
            round(ys.max() / cfg.IMAGE_HEIGHT * 1000),
            round(xs.max() / cfg.IMAGE_WIDTH * 1000),
        ],
    }


def test_physical_pipeline():
    """렌더링·역투영·실제 집기·중복 실행 방지를 검증합니다."""
    robot, viewer = prepare_robot()
    robot.hold(viewer, 1.5)
    agent = Agent()

    # 테스트 이미지와 JSON은 임시 폴더에만 저장합니다.
    with (
        tempfile.TemporaryDirectory() as folder,
        patch.object(cfg, "OUTPUT_DIR", Path(folder)),
        mujoco.Renderer(
            robot.model,
            height=cfg.IMAGE_HEIGHT,
            width=cfg.IMAGE_WIDTH,
        ) as renderer,
    ):
        image, depth = capture(renderer, robot)
        args = synthetic_selection(image)
        _, source, cube_name, cube_id = validate_selection(
            args, robot, depth
        )
        assert cube_name == "blue_cube"
        assert cube_id == robot.cube_ids["blue_cube"]

        error = float(np.linalg.norm(
            source - robot.data.xpos[cube_id]
        ))
        assert error < 0.004, f"역투영 오차: {error:.6f}m"

        result = agent.execute(
            robot, viewer, renderer,
            image, depth, "offline test",
            args, "selftest",
        )
        assert result["status"] == "success", result

        image2, depth2 = capture(renderer, robot)
        args2 = synthetic_selection(image2)
        _, source2, _, cube_id2 = validate_selection(args2, robot, depth2)
        assert cube_id2 == cube_id

        repeated = robot.execute_pick(viewer, source2, cube_id2)
        assert repeated["status"] == "already_at_destination"

        try:
            validate_selection(
                {"label": "floor", "box_2d": [10, 10, 30, 30]},
                robot, depth2,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("바닥 선택이 거부되지 않았습니다.")

        print(
            "PASS 물리 파이프라인:",
            result["evaluation_after_parking"],
        )


def test_unreachable_target():
    """실제 집게가 목표에 도달하지 못하면 예외가 나는지 확인합니다."""
    robot, viewer = prepare_robot()
    robot.hold(viewer, 1.5)

    target = (
        robot.data.site_xpos[robot.site_id].copy()
        + np.array([0.5, 0, 0])
    )

    try:
        robot.wait_reached(viewer, target, timeout=0.02)
    except RuntimeError as exc:
        assert "도달 실패" in str(exc)
    else:
        raise AssertionError("도달 불가능한 목표가 통과됐습니다.")

    print("PASS 실제 도달 실패")


def test_mid_motion_stop():
    """이동 중 중단 신호가 발생하면 해당 스텝에서 멈추는지 확인합니다."""
    robot, viewer = prepare_robot()
    started = robot.data.time

    def trigger_stop():
        """시뮬레이션 0.05초에 사용자 중단을 흉내 냅니다."""
        if robot.data.time - started >= 0.05:
            robot.cancel_event.set()

    viewer.sync = trigger_stop
    target = robot.data.ctrl.copy()
    target[robot.arm_ids[0]] += 0.1

    try:
        robot.move_controls(viewer, target, 2.0)
    except OperationCancelled:
        assert robot.data.time - started <= 0.054
    else:
        raise AssertionError("이동 중 중단 신호가 무시됐습니다.")

    # 중단 처리로 집게 목표가 바뀌지 않는지도 검사합니다.
    grip_before = robot.data.ctrl[robot.grip_id]
    robot.hold_current_arm()

    assert robot.data.ctrl[robot.grip_id] == grip_before
    np.testing.assert_allclose(
        robot.data.ctrl[robot.arm_ids],
        np.clip(
            robot.data.qpos[robot.qpos_ids],
            robot.lower,
            robot.upper,
        ),
    )

    print("PASS 이동 중 중단")


def test_failure_blocks_next_stage():
    """접근 도달 실패 뒤 집게 닫기가 실행되지 않는지 확인합니다."""
    robot, viewer = prepare_robot()
    robot.hold(viewer, 1.5)
    cube_id = robot.cube_ids["blue_cube"]
    source = robot.data.xpos[cube_id].copy()

    with (
        patch.object(
            robot, "wait_reached",
            side_effect=RuntimeError("injected arrival failure"),
        ),
        patch.object(
            robot, "set_gripper",
            wraps=robot.set_gripper,
        ) as gripper,
    ):
        try:
            robot.execute_pick(viewer, source, cube_id)
        except RuntimeError as exc:
            assert "injected arrival failure" in str(exc)
        else:
            raise AssertionError("도달 실패가 무시됐습니다.")

        # 첫 집게 열기만 실행되고 집기용 닫기는 실행되지 않아야 합니다.
        assert gripper.call_count == 1
        assert robot.stage == "approach"

    print("PASS 실패 후 후속 단계 차단")


def test_viewer_closed():
    """닫힌 뷰어에서 동작 완료로 잘못 처리하지 않는지 확인합니다."""
    robot, viewer = prepare_robot()
    viewer.running = False

    try:
        robot.hold(viewer, 1.0)
    except OperationCancelled:
        pass
    else:
        raise AssertionError("닫힌 뷰어가 중단으로 처리되지 않았습니다.")

    print("PASS 뷰어 종료")


def test_failure_latch():
    """작업 오류 뒤 실패 상태가 고정되고 다음 이동이 거부되는지 확인합니다."""
    robot, viewer = prepare_robot()
    agent = Agent()

    with (
        tempfile.TemporaryDirectory() as folder,
        patch.object(cfg, "OUTPUT_DIR", Path(folder)),
        patch(
            "codes.agent.validate_selection",
            side_effect=RuntimeError("injected failure"),
        ),
        patch(
            "codes.agent.capture",
            side_effect=RuntimeError("injected rendering failure"),
        ),
    ):
        result = agent.execute(
            robot, viewer, None,
            None, None, "test", {}, "failure",
        )

    assert result["status"] == "error"
    assert robot.faulted
    assert "image_save_error" in result

    cube_id = robot.cube_ids["blue_cube"]

    try:
        robot.execute_pick(viewer, cfg.PICK.copy(), cube_id)
    except RuntimeError as exc:
        assert "초기화" in str(exc)
    else:
        raise AssertionError("실패 후 새 작업이 허용됐습니다.")

    print("PASS 실패 상태 유지")


def run_tests():
    """API 없는 통합 및 실패 처리 검증을 순서대로 실행합니다."""
    tests = [
        test_physical_pipeline,
        test_unreachable_target,
        test_mid_motion_stop,
        test_failure_blocks_next_stage,
        test_viewer_closed,
        test_failure_latch,
    ]

    for test in tests:
        test()

    print(f"\n전체 {len(tests)}개 테스트 통과")
    print("Gemini API 인식 성능은 이 테스트에 포함되지 않습니다.")


if __name__ == "__main__":
    run_tests()