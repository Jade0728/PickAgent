"""실제 Gemini와 MuJoCo로 2배치 × 3색 정상 요청을 검증합니다."""

import json
from datetime import datetime

import mujoco
from google import genai

from codes import config as cfg
from codes.agent import Agent, capture
from codes.test_pipeline import prepare_robot


REQUESTS = (
    ("blue_cube", "파란 굿즈 뽑아줘"),
    ("red_cube", "빨간 굿즈 하나 줘"),
    ("yellow_cube", "노란 굿즈 뽑아줘"),
)


def run_case(agent, layout, expected_cube, request, case_id):
    robot, viewer = prepare_robot(layout=layout)
    robot.hold(viewer, 1.5)
    expected_id = robot.cube_ids[expected_cube]

    row = {
        "case": case_id,
        "layout": layout,
        "request": request,
        "expected_cube": expected_cube,
        "expected_result": "요청한 큐브가 배출구에 도착",
        "passed": False,
    }

    with mujoco.Renderer(
        robot.model,
        height=cfg.IMAGE_HEIGHT,
        width=cfg.IMAGE_WIDTH,
    ) as renderer:
        image, depth = capture(renderer, robot)
        image.save(cfg.OUTPUT_DIR / f"{case_id}_before.png")

        _, response = agent.select(image, request)
        calls = response.function_calls or []

        if len(calls) != 1 or calls[0].name != "pick_and_place":
            row["status"] = "unexpected_tool_response"
            row["message"] = response.text or "단일 pick_and_place 호출이 필요합니다."
            row["tool_calls"] = [call.name for call in calls]
            return row

        args = dict(calls[0].args or {})
        result = agent.execute(
            robot, viewer, renderer, image, depth, request, args, case_id
        )

    expected_arrival = robot.evaluate_destination(expected_id)
    other_cubes_at_destination = [
        name
        for name, cube_id in robot.cube_ids.items()
        if name != expected_cube
        and robot.evaluate_destination(cube_id)["success"]
    ]

    row.update(
        function_args=args,
        status=result["status"],
        selected_cube=result.get("selected_cube"),
        stage=result.get("stage"),
        message=result["message"],
        expected_cube_arrival=expected_arrival,
        other_cubes_at_destination=other_cubes_at_destination,
    )
    row["passed"] = (
        result["status"] == "success"
        and result.get("selected_cube") == expected_cube
        and expected_arrival["success"]
        and not other_cubes_at_destination
    )
    return row


def main():
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    summary_path = cfg.OUTPUT_DIR / f"normal_{run_id}_summary.json"
    rows = []

    with genai.Client(api_key=cfg.load_api_key()) as client:
        agent = Agent(client)

        for layout in ("A", "B"):
            for expected_cube, request in REQUESTS:
                case_id = f"normal_{run_id}_{layout}_{expected_cube}"

                try:
                    row = run_case(
                        agent, layout, expected_cube, request, case_id
                    )
                except Exception as exc:
                    row = {
                        "case": case_id,
                        "layout": layout,
                        "request": request,
                        "expected_cube": expected_cube,
                        "expected_result": "요청한 큐브가 배출구에 도착",
                        "status": "execution_error",
                        "message": f"{type(exc).__name__}: {exc}",
                        "passed": False,
                    }

                rows.append(row)
                summary_path.write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                verdict = "PASS" if row["passed"] else "FAIL"
                print(f"[{verdict}] 배치 {layout} / {request} / {row['message']}")

    passed = sum(row["passed"] for row in rows)
    print(f"\n정상 요청 {passed}/{len(rows)}건 통과")
    print(f"결과 파일: {summary_path}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())