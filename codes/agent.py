# 역할: 영상 촬영·좌표 변환·Gemini 통신·작업 실행 결과 기록을 담당합니다.

import io
import json
import math
from unittest import result

import mujoco
import numpy as np
from google.genai import types
from PIL import Image, ImageDraw

from . import config as cfg
from .robot import OperationCancelled, find_id
from codes import robot


def capture(renderer, robot):
    """같은 카메라에서 RGB와 m 단위 깊이 영상을 얻습니다."""
    mujoco.mj_forward(robot.model, robot.data)

    renderer.disable_depth_rendering()
    renderer.update_scene(
        robot.data, camera=cfg.CAMERA_NAME
    )
    image = Image.fromarray(renderer.render().copy())

    renderer.enable_depth_rendering()
    try:
        renderer.update_scene(
            robot.data, camera=cfg.CAMERA_NAME
        )
        depth = renderer.render().copy()
    finally:
        renderer.disable_depth_rendering()

    return image, depth


def pixel_to_world(robot, u, v, depth):
    """카메라 파라미터와 깊이로 픽셀을 세계 좌표로 역투영합니다."""
    camera_id = find_id(
        robot.model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        cfg.CAMERA_NAME,
    )
    focal = cfg.IMAGE_HEIGHT / (
        2 * math.tan(
            math.radians(robot.model.cam_fovy[camera_id]) / 2
        )
    )

    camera_point = np.array([
        (u + 0.5 - cfg.IMAGE_WIDTH / 2) / focal,
        -(v + 0.5 - cfg.IMAGE_HEIGHT / 2) / focal,
        -1.0,
    ]) * depth

    rotation = robot.data.cam_xmat[camera_id].reshape(3, 3)
    return robot.data.cam_xpos[camera_id] + rotation @ camera_point


def validate_selection(args, robot, depth):
    """도구 인자를 검증하고 선택한 큐브의 파지 좌표를 계산합니다."""
    if set(args) != {"label", "box_2d"}:
        raise ValueError("함수 인자는 label과 box_2d여야 합니다.")

    if not isinstance(args["label"], str) or not args["label"].strip():
        raise ValueError("물체 이름이 비어 있습니다.")

    box = args["box_2d"]
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("박스 좌표는 네 개여야 합니다.")

    if any(
        type(value) not in (int, float)
        or not 0 <= value <= 1000
        or int(value) != value
        for value in box
    ):
        raise ValueError("좌표는 0~1000 정수여야 합니다.")

    y1, x1, y2, x2 = box
    if not (x1 < x2 and y1 < y2):
        raise ValueError("박스 좌표 순서가 잘못됐습니다.")

    width, height = cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT
    pixels = [
        min(width - 1, round(x1 * width / 1000)),
        min(height - 1, round(y1 * height / 1000)),
        min(width - 1, round(x2 * width / 1000)),
        min(height - 1, round(y2 * height / 1000)),
    ]

    if pixels[0] >= pixels[2] or pixels[1] >= pixels[3]:
        raise ValueError("픽셀 변환 후 박스가 너무 작습니다.")

    u = (pixels[0] + pixels[2]) // 2
    v = (pixels[1] + pixels[3]) // 2

    sample = depth[
        max(0, v - 1):min(height, v + 2),
        max(0, u - 1):min(width, u + 2),
    ]
    z_depth = float(np.median(sample))

    if not math.isfinite(z_depth) or z_depth <= 0:
        raise ValueError("선택 지점의 깊이가 유효하지 않습니다.")

    surface = pixel_to_world(robot, u, v, z_depth)

    # 현재 데모는 똑바로 선 4cm 큐브만 지원합니다.
    if not 0.032 <= surface[2] <= 0.048:
        raise ValueError("선택점이 4cm 큐브의 상면이 아닙니다.")

    point = surface.copy()
    point[2] -= cfg.CUBE_HALF

    if not (
        0.13 <= point[0] <= 0.28
        and -0.10 <= point[1] <= 0.14
    ):
        raise ValueError("선택 위치가 작업 영역 밖입니다.")

    matches = []
    tolerance_m = 0.003

    for cube_name, cube_id in robot.cube_ids.items():
        center = robot.data.xpos[cube_id]
        rotation = robot.data.xmat[cube_id].reshape(3, 3)
        local = rotation.T @ (surface - center)

        if (
            np.all(np.abs(local[:2]) <= cfg.CUBE_HALF + tolerance_m)
            and abs(local[2] - cfg.CUBE_HALF) <= tolerance_m
        ):
            matches.append((cube_name, cube_id))

    if len(matches) != 1:
        raise ValueError("인식 위치를 장면의 큐브 하나에 연결할 수 없습니다.")

    cube_name, cube_id = matches[0]
    return pixels, point, cube_name, cube_id


class Agent:
    """Gemini와 로봇 사이에서 인식·검증·기록을 연결합니다."""

    def __init__(self, client=None):
        """오프라인 테스트에서는 API 클라이언트 없이 사용할 수 있습니다."""
        self.client = client
        self.tool = types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="pick_and_place",
                description="선택한 4cm 큐브를 고정 수령 위치로 옮깁니다.",
                parameters={
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "box_2d": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": (
                                "[ymin,xmin,ymax,xmax], integers 0-1000"
                            ),
                        },
                    },
                    "required": ["label", "box_2d"],
                },
            )
        ])

    def select(self, image, request):
        """이번 이미지와 요청만 사용해 Gemini에 대상 선택을 요청합니다."""
        if self.client is None:
            raise RuntimeError("Gemini 클라이언트가 없습니다.")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        history = [types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    data=buffer.getvalue(),
                    mime_type="image/png",
                ),
                types.Part.from_text(text=request),
            ],
        )]

        response = self.client.models.generate_content(
            model=cfg.MODEL_NAME,
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=cfg.SYSTEM_INSTRUCTION,
                tools=[self.tool],
                automatic_function_calling=(
                    types.AutomaticFunctionCallingConfig(disable=True)
                ),
            ),
        )
        return history, response

    def explain(self, history, response, call, result):
        """실행 결과를 전달하고 추가 도구 호출 없이 안내를 받습니다."""
        answer = types.FunctionResponse(
            name=call.name,
            response=result,
        )
        if call.id:
            answer.id = call.id

        contents = list(history)
        contents.append(response.candidates[0].content)
        contents.append(types.Content(
            role="user",
            parts=[types.Part(function_response=answer)],
        ))

        final = self.client.models.generate_content(
            model=cfg.MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=cfg.SYSTEM_INSTRUCTION,
                tools=[self.tool],
                tool_config=types.ToolConfig(
                    function_calling_config=(
                        types.FunctionCallingConfig(mode="NONE")
                    )
                ),
                automatic_function_calling=(
                    types.AutomaticFunctionCallingConfig(disable=True)
                ),
            ),
        )
        return final.text or result["message"]

    def execute(
        self, robot, viewer, renderer,
        image, depth, request, args, prefix,
    ):
        """선택 검증부터 물리 실행·실패 차단·결과 저장까지 처리합니다."""
        result = {"status": "error", "message": "미실행"}
        cube_name = None

        try:
            robot.check_operation(viewer)
            pixels, source, cube_name, cube_id = validate_selection(args, robot, depth)

            cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            marked = image.copy()
            ImageDraw.Draw(marked).rectangle(
                pixels, outline="red", width=3
            )
            marked.save(cfg.OUTPUT_DIR / f"{prefix}_box.png")

            result = robot.execute_pick(viewer, source, cube_id)

            if result["status"] == "success":
                robot.park(viewer)

                evaluation = robot.evaluate_destination(cube_id)
                result["evaluation_after_parking"] = evaluation

                if not evaluation["success"]:
                    result.update(
                        status="failed",
                        message="팔 복귀 후 목적지 위치 검증 실패",
                    )

            robot.check_operation(viewer)

        except OperationCancelled as exc:
            result = {"status": "cancelled", "message": str(exc)}

        except Exception as exc:
            result = {"status": "error", "message": str(exc)}

        finally:
            if cube_name is not None:
                result["selected_cube"] = cube_name
                
            result["stage"] = robot.stage
            result["reach_metrics"] = robot.last_reach_metrics

            if result["status"] not in {
                "success", "already_at_destination"
            }:
                robot.faulted = True
                robot.hold_current_arm()

            # 이미지 저장 실패와 로봇 작업 결과를 구분합니다.
            try:
                cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                after, _ = capture(renderer, robot)
                after.save(
                    cfg.OUTPUT_DIR / f"{prefix}_after.png"
                )
            except Exception as exc:
                result["image_save_error"] = str(exc)

            try:
                record = {
                    "request": request,
                    "function_args": args,
                    "result": result,
                }
                (
                    cfg.OUTPUT_DIR / f"{prefix}_result.json"
                ).write_text(
                    json.dumps(
                        record, ensure_ascii=False, indent=2
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:
                result["record_save_error"] = str(exc)

        print(result["message"])
        return result