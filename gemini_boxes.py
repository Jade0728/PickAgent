# 상품별 좌표를 JSON으로 받고, 사진 위에 박스를 그려서 저장하는 예제입니다.
import io
import json
import os
import sys
from pathlib import Path 
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel, Field


class DetectedObject(BaseModel):
    label: str = Field(description="Short English product name")
    box_2d: list[int] = Field(
        description=(
            "Exactly four integers: "
            "[y_min, x_min, y_max, x_max], normalized to 0-1000"
        )
    )


class DetectionResult(BaseModel):
    objects: list[DetectedObject]


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("CMD에서 GEMINI_API_KEY를 먼저 설정하세요.")

    if len(sys.argv) != 2:
        raise SystemExit(
            '사용법: python gemini_boxes.py "images/products.png"'
        )

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
  
        raise SystemExit(f"사진 파일을 찾을 수 없습니다: {image_path}")

    # 휴대전화 사진의 회전 정보를 적용한 동일한 이미지를
    # Gemini 전송과 박스 그리기에 모두 사용합니다.
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    width, height = image.size

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")

    image_part = types.Part.from_bytes(
        data=buffer.getvalue(),
        mime_type="image/jpeg",
    )

    prompt = """
Detect every separate product on the tabletop.

For each product:
- Return a short English label, such as "blue teddy bear".
- Return a tight bounding box around the entire visible product.
- Use box_2d = [y_min, x_min, y_max, x_max].
- Normalize coordinates to integers from 0 to 1000,
  relative to the full input image.
- Do not include shadows, the table, or the background.
- Do not return duplicate boxes for parts of one product.

If there are no products, return an empty objects list.
"""

    print("Gemini에서 상품 좌표를 받는 중입니다...")

    with genai.Client(api_key=api_key) as client:
        response = client.models.generate_content(
            model="gemini-robotics-er-2-preview",
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DetectionResult,
            ),
        )

    if not response.text:
        raise SystemExit("텍스트 응답이 없습니다.")

    # JSON 형식과 필드 타입을 검증합니다.
    result = DetectionResult.model_validate_json(response.text)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # 원본 사진을 덮어쓰지 않고 별도 파일로 저장합니다.
    prefix = image_path.stem
    raw_path = output_dir / f"{prefix}_detections.json"
    raw_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(16, width // 65))
    line_width = max(3, width // 350)
    colors = ["red", "blue", "green", "orange", "purple"]

    pixel_results = []

    for index, obj in enumerate(result.objects, start=1):
        box = obj.box_2d

        # JSON 스키마만으로 보장되지 않는 좌표 조건도 확인합니다.
        if len(box) != 4:
            print(f"건너뜀: {obj.label} - 좌표가 4개가 아닙니다.")
            continue

        y_min, x_min, y_max, x_max = box

        if not (
            0 <= x_min < x_max <= 1000
            and 0 <= y_min < y_max <= 1000
        ):
            print(f"건너뜀: {obj.label} - 잘못된 좌표 {box}")
            continue

        # 정규화 좌표를 픽셀로 변환합니다.
        x1 = min(width - 1, round(x_min / 1000 * width))
        y1 = min(height - 1, round(y_min / 1000 * height))
        x2 = min(width - 1, round(x_max / 1000 * width))
        y2 = min(height - 1, round(y_max / 1000 * height))

        if x1 >= x2 or y1 >= y2:
            print(f"건너뜀: {obj.label} - 픽셀 변환 후 박스가 너무 작습니다.")
            continue

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        color = colors[(index - 1) % len(colors)]

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=line_width,
        )

        radius = max(4, width // 200)
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill=color,
        )

        # 영문 라벨을 사용하므로 별도 한글 폰트가 필요하지 않습니다.
        draw.text(
            (x1 + 5, y1 + 5),
            f"{index}: {obj.label}",
            font=font,
            fill=color,
            stroke_width=2,
            stroke_fill="white",
        )

        pixel_results.append({
            "label": obj.label,
            "box_xyxy_pixels": [x1, y1, x2, y2],
            "center_xy_pixels": [center_x, center_y],
        })

        print(
            f"{index}. {obj.label}\n"
            f"   정규화 [ymin, xmin, ymax, xmax]: {box}\n"
            f"   픽셀 [x1, y1, x2, y2]: {[x1, y1, x2, y2]}\n"
            f"   중심점 (x, y): ({center_x}, {center_y})"
        )

    image_output = output_dir / f"{prefix}_boxes.png"
    image.save(image_output)

    pixel_output = output_dir / f"{prefix}_pixels.json"
    pixel_output.write_text(
        json.dumps(
            {
                "image_size": {"width": width, "height": height},
                "objects": pixel_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n박스 이미지: {image_output.resolve()}")
    print(f"정규화 좌표: {raw_path.resolve()}")
    print(f"픽셀 좌표: {pixel_output.resolve()}")


if __name__ == "__main__":
    main()