# 역할: 간단한 터미널 입력으로 작업을 실행하고 중단·초기화·종료를 관리합니다.

import argparse
import queue
import threading
import time

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from uuid import uuid4

import mujoco
import mujoco.viewer
import numpy as np

from google import genai
from google.genai import types


# 직접 실행과 패키지 실행을 모두 지원합니다.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(
        0, str(Path(__file__).resolve().parent.parent)
    )

from codes import config as cfg
from codes.agent import Agent, capture
from codes.robot import OperationCancelled, Robot


def configure_viewer(viewer):
    """로봇을 관찰하기 좋은 시점으로 설정합니다."""
    with viewer.lock():
        viewer.cam.lookat[:] = [0.15, 0, 0.15]
        viewer.cam.distance = 0.85
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25


def clear_queue(items):
    """중단 전에 대기하던 항목을 폐기합니다."""
    while True:
        try:
            items.get_nowait()
        except queue.Empty:
            return


def input_worker(
    commands,
    notices,
    accepting,
    cancel_event,
    exit_event,
):
    """입력만 담당하며 입력 안내 문구는 출력하지 않습니다."""
    while True:
        try:
            text = input().strip()
        except EOFError:
            text = "/종료"

        if not text:
            continue

        command = text.lower()

        # 중단·종료는 작업 완료를 기다리는 큐를 거치지 않습니다.
        if command in {"/종료", "종료", "/quit", "/exit"}:
            accepting.clear()
            exit_event.set()
            cancel_event.set()
            return

        if command in {"/중단", "중단", "/stop"}:
            accepting.clear()
            cancel_event.set()
            continue

        # 작업 중 일반 요청과 초기화 요청은 쌓지 않습니다.
        if not accepting.is_set():
            notices.put(
                "[안내] 작업 중 입력은 실행하지 않습니다. "
                "/중단 또는 /종료를 사용하세요."
            )
            continue

        accepting.clear()

        if command in {"/초기화", "초기화", "/reset"}:
            commands.put(("reset", ""))
        else:
            commands.put(("run", text))


def run_interactive():
    """입력·비동기 Gemini 호출·로봇 동작을 연결합니다."""
    key = cfg.load_api_key()

    cancel_event = threading.Event()
    exit_event = threading.Event()
    accepting = threading.Event()

    commands = queue.Queue()
    notices = queue.Queue()

    robot = Robot(cancel_event)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSO-101 Pick Agent")
    print("명령: /중단  /초기화  /종료")
    print("작업 중에는 /중단, /종료만 처리합니다.")
    print("시뮬레이션을 준비하고 있습니다.")

    with (
        genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=60000),
        ) as client,
        mujoco.Renderer(
            robot.model,
            height=cfg.IMAGE_HEIGHT,
            width=cfg.IMAGE_WIDTH,
        ) as renderer,
        mujoco.viewer.launch_passive(
            robot.model, robot.data
        ) as viewer,
    ):
        configure_viewer(viewer)
        robot.hold(viewer, 1.5)

        agent = Agent(client)
        pool = ThreadPoolExecutor(max_workers=1)

        pending = None
        phase = None
        context = None
        prompt_shown = False

        threading.Thread(
            target=input_worker,
            args=(
                commands,
                notices,
                accepting,
                cancel_event,
                exit_event,
            ),
            daemon=True,
        ).start()

        try:
            while viewer.is_running() and not exit_event.is_set():
                try:
                    # 입력 스레드는 출력 대신 안내 메시지를 큐에 넣습니다.
                    while True:
                        try:
                            print(notices.get_nowait())
                        except queue.Empty:
                            break

                    if cancel_event.is_set():
                        accepting.clear()
                        robot.hold_current_arm()
                        robot.faulted = True

                        if pending is not None:
                            pending.cancel()

                        # 진행 중인 HTTP 요청의 결과로 동작하지 않게 폐기합니다.
                        pending = None
                        phase = None
                        context = None

                        clear_queue(commands)
                        cancel_event.clear()
                        prompt_shown = False

                        print(
                            "\n[중단] 후속 이동을 차단했습니다. "
                            "/초기화 후 다시 요청하세요."
                        )

                    if pending is None:
                        # 작업 중에는 다음 입력 안내를 출력하지 않습니다.
                        if not prompt_shown:
                            label = (
                                "초기화 또는 종료"
                                if robot.faulted
                                else "요청"
                            )
                            print(
                                f"\n{label} > ",
                                end="",
                                flush=True,
                            )
                            accepting.set()
                            prompt_shown = True

                        try:
                            command, text = commands.get_nowait()
                        except queue.Empty:
                            robot.step(viewer)
                            continue

                        accepting.clear()
                        prompt_shown = False

                        if command == "reset":
                            print("[초기화] 장면을 되돌리고 있습니다.")
                            robot.reset(viewer)
                            print("[완료] 새 요청을 입력할 수 있습니다.")
                            continue

                        if robot.faulted:
                            print(
                                "[차단] 먼저 /초기화를 입력하세요."
                            )
                            continue

                        robot.stage = "capture"
                        robot.last_reach_metrics = None

                        prefix = (
                            datetime.now().strftime("%Y%m%d_%H%M%S")
                            + "_"
                            + uuid4().hex[:6]
                        )

                        image, depth = capture(renderer, robot)
                        image.save(
                            cfg.OUTPUT_DIR / f"{prefix}_before.png"
                        )

                        context = (
                            text,
                            prefix,
                            image,
                            depth,
                            robot.data.qpos.copy(),
                        )

                        print(
                            "[분석 중] 카메라 이미지에서 "
                            "대상을 찾고 있습니다."
                        )

                        pending = pool.submit(
                            agent.select,
                            image,
                            text,
                        )
                        phase = "selection"

                    elif pending.done():
                        if phase == "final":
                            # 안내 API 실패와 로봇 동작 결과를 구분합니다.
                            try:
                                print(
                                    "[Agent]",
                                    pending.result(),
                                )
                            except Exception as exc:
                                print(
                                    "[안내 실패] 동작 결과는 저장됐습니다.",
                                    exc,
                                )

                            pending = None
                            context = None
                            continue

                        history, response = pending.result()
                        calls = response.function_calls or []

                        if not calls:
                            print(
                                "[Agent]",
                                response.text or "응답이 없습니다.",
                            )
                            pending = None
                            context = None
                            continue

                        if (
                            len(calls) != 1
                            or calls[0].name != "pick_and_place"
                        ):
                            raise ValueError(
                                "지원되는 단일 함수 호출이 아닙니다."
                            )

                        (
                            request,
                            prefix,
                            image,
                            depth,
                            snapshot,
                        ) = context

                        robot.check_operation(viewer)

                        # 촬영 후 장면이 바뀌면 이전 이미지 좌표를 거부합니다.
                        if not np.allclose(
                            robot.data.qpos,
                            snapshot,
                            atol=1e-5,
                            rtol=0,
                        ):
                            raise ValueError(
                                "촬영 후 장면이 변경됐습니다."
                            )

                        call = calls[0]
                        args = dict(call.args or {})

                        print(
                            "[선택 완료]",
                            args.get("label", "이름 없음"),
                        )

                        # 이동 중에도 입력 스레드가 중단 이벤트를 설정합니다.
                        # robot.py가 각 제어 스텝과 IK 반복에서 확인합니다.
                        result = agent.execute(
                            robot,
                            viewer,
                            renderer,
                            image,
                            depth,
                            request,
                            args,
                            prefix,
                        )
                        pending = None

                        if robot.faulted:
                            print(
                                "[차단] /초기화 후 다시 요청하세요."
                            )
                            continue

                        robot.check_operation(viewer)

                        evaluation = (
                            result.get("evaluation_after_parking")
                            or result.get("evaluation")
                        )
                        if evaluation:
                            print(
                                "[결과] 수평 오차 "
                                f"{evaluation['xy_error_mm']:.1f}mm"
                            )

                        print("[안내 생성 중] 실행 결과를 정리합니다.")

                        pending = pool.submit(
                            agent.explain,
                            history,
                            response,
                            call,
                            result,
                        )
                        phase = "final"

                    else:
                        # API 대기 중에는 촬영 시점의 물리 상태를 유지합니다.
                        viewer.sync()
                        time.sleep(0.01)

                except OperationCancelled:
                    if not viewer.is_running():
                        print("\n[종료] MuJoCo 창이 닫혔습니다.")
                        break

                    cancel_event.set()

                except Exception as exc:
                    accepting.clear()
                    robot.faulted = True
                    robot.hold_current_arm()

                    if pending is not None:
                        pending.cancel()

                    pending = None
                    phase = None
                    context = None
                    prompt_shown = False

                    clear_queue(commands)

                    print(
                        f"\n[실패] {exc}\n"
                        "/초기화 후 다시 요청하세요."
                    )

        finally:
            accepting.clear()
            robot.hold_current_arm()

            if pending is not None:
                pending.cancel()

            print(
                "\n[종료] 자원을 정리합니다. "
                "진행 중인 API 통신이 있으면 응답 정리를 기다립니다."
            )

            # API가 끝난 뒤 클라이언트와 렌더러를 닫습니다.
            pool.shutdown(
                wait=True,
                cancel_futures=True,
            )


def run_demo():
    """Gemini 없이 고정 큐브를 옮기는 기존 데모입니다."""
    robot = Robot()

    with mujoco.viewer.launch_passive(
        robot.model, robot.data
    ) as viewer:
        configure_viewer(viewer)

        try:
            robot.hold(viewer, 1.5)

            # 독립 데모에서만 시뮬레이터 정답 좌표를 사용합니다.
            cube_id = robot.cube_ids["blue_cube"]
            source = robot.data.xpos[cube_id].copy()
            result = robot.execute_pick(viewer, source, cube_id)
            print(result["message"])

            if result["status"] == "success":
                robot.park(viewer)
                print(robot.evaluate_destination(cube_id))

            print("확인이 끝나면 MuJoCo 창을 닫으세요.")

            while viewer.is_running():
                robot.step(viewer)

        except OperationCancelled:
            print("[중단] 데모를 종료합니다.")

        finally:
            robot.hold_current_arm()


def main():
    """터미널 실행·독립 데모·통합 테스트 중 하나를 시작합니다."""
    parser = argparse.ArgumentParser(
        description="SO-101 Pick Agent 터미널 버전"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--demo",
        action="store_true",
        help="API 없는 독립 데모",
    )
    group.add_argument(
        "--self-test",
        action="store_true",
        help="API 없는 통합 테스트",
    )
    options = parser.parse_args()

    try:
        if options.self_test:
            from codes.test_pipeline import run_tests

            run_tests()
        elif options.demo:
            run_demo()
        else:
            run_interactive()

    except KeyboardInterrupt:
        print("\n[종료] 키보드에서 실행을 중단했습니다.")

    except Exception as exc:
        print(f"\n[오류] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()