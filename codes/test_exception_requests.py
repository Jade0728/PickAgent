"""Gemini와 MuJoCo로 팀 공통 예외 요청 4건을 검증합니다."""

import json
import time
from datetime import datetime
from unittest.mock import patch

import mujoco
from google import genai

from codes import config as cfg
from codes.agent import Agent, capture
from codes.test_pipeline import prepare_robot


CASES = (
    ("missing", "보라색 굿즈 뽑아줘", "대상을 찾지 못했다고 안내"),
    ("ambiguous", "빨간 굿즈 하나 줘", "두 빨간 큐브 중 어느 것인지 재질문"),
    ("out_of_range", "파란 굿즈 뽑아줘", "작업 범위 밖으로 거절"),
    ("grasp_failure", "파란 굿즈 뽑아줘", "파지 실패 기록 후 초기화·재실행 성공"),
)


def prepare_case(case):
    if case == "ambiguous":
        colors = {
            **cfg.CUBE_COLORS,
            "yellow_cube": cfg.CUBE_COLORS["red_cube"],
        }
        with patch.object(cfg, "CUBE_COLORS", colors):
            return prepare_robot(layout="A")

    if case == "out_of_range":
        slots = ((0.30, 0.00), *cfg.CUBE_SLOTS_XY[1:])
        with patch.object(cfg, "CUBE_SLOTS_XY", slots):
            return prepare_robot(layout="A")

    return prepare_robot(layout="A")


def run_case(agent, case, request, prefix, row):
    robot, viewer = prepare_case(case)
    robot.hold(viewer, 1.5)

    with mujoco.Renderer(
        robot.model,
        height=cfg.IMAGE_HEIGHT,
        width=cfg.IMAGE_WIDTH,
    ) as renderer:
        image, depth = capture(renderer, robot)
        image.save(cfg.OUTPUT_DIR / f"{prefix}_before.png")

        _, response = agent.select(image, request)
        calls = response.function_calls or []
        row["tool_calls"] = [call.name for call in calls]

        # 대상 없음·모호함에서 도구를 호출하면 실패로 기록하고 실행을 차단합니다.
        if case in {"missing", "ambiguous"}:
            row["arm_moved"] = False
            row["message"] = response.text or "텍스트 안내 없음"

            if calls:
                row["message"] = "예상하지 않은 도구 호출이 반환됨"
                return

            print(f"\nGemini 응답: {row['message']}")
            print(f"기대 결과: {row['expected']}")
            row["passed"] = input("기대 결과와 일치하면 y: ").strip().lower() == "y"
            return

        if len(calls) != 1 or calls[0].name != "pick_and_place":
            row["message"] = response.text or "단일 pick_and_place 호출이 필요함"
            return

        args = dict(calls[0].args or {})
        row["function_args"] = args

        if case == "out_of_range":
            with patch.object(
                robot, "move_controls", wraps=robot.move_controls
            ) as motion:
                result = agent.execute(
                    robot, viewer, renderer, image, depth, request, args, prefix
                )

            row["motion_calls"] = motion.call_count
            row["result"] = result
            row["passed"] = (
                result["status"] == "error"
                and "작업 영역 밖" in result["message"]
                and motion.call_count == 0
            )
            return

        row["failure_condition"] = "GRIP_CLOSE를 GRIP_OPEN과 같은 각도로 설정"
        with patch.object(cfg, "GRIP_CLOSE", cfg.GRIP_OPEN):
            result = agent.execute(
                robot, viewer, renderer, image, depth, request, args, prefix
            )

        row["result"] = result
        row["faulted_before_reset"] = robot.faulted
        expected_failure = (
            result["status"] == "error"
            and result.get("selected_cube") == "blue_cube"
            and result.get("stage") == "lift"
            and "파지 실패" in result["message"]
            and robot.faulted
        )
        if not expected_failure:
            return

        # 닫힘 각도를 복구한 뒤 초기화하고, 같은 Gemini 선택을 재사용합니다.
        robot.reset(viewer)
        robot.hold(viewer, 1.5)
        retry_image, retry_depth = capture(renderer, robot)
        retry = agent.execute(
            robot, viewer, renderer, retry_image, retry_depth,
            request, args, f"{prefix}_retry",
        )
        row["retry_after_reset"] = retry
        row["passed"] = (
            retry["status"] == "success"
            and retry.get("selected_cube") == "blue_cube"
            and robot.evaluate_destination(robot.cube_ids["blue_cube"])["success"]
        )


def main():
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = cfg.OUTPUT_DIR / f"exception_{run_id}_summary.json"
    rows = []

    with genai.Client(api_key=cfg.load_api_key()) as client:
        agent = Agent(client)

        for case, request, expected in CASES:
            if rows:
                time.sleep(13)

            prefix = f"exception_{run_id}_{case}"
            row = {
                "case": case,
                "request": request,
                "expected": expected,
                "passed": False,
            }
            try:
                run_case(agent, case, request, prefix, row)
            except Exception as exc:
                row["execution_error"] = f"{type(exc).__name__}: {exc}"

            rows.append(row)
            summary.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            verdict = "PASS" if row["passed"] else "FAIL"
            detail = (
                row.get("execution_error")
                or row.get("message")
                or row.get("result", {}).get("message", "")
            )
            print(f"[{verdict}] {case} / {detail}")

    passed = sum(row["passed"] for row in rows)
    print(f"\n예외 요청 {passed}/{len(rows)}건 통과")
    print(f"결과 파일: {summary}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())