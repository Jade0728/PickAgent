# gemini api를 사용하여 사진을 분석하는 예제
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("CMD에서 GEMINI_API_KEY를 먼저 설정하세요.")

    if len(sys.argv) != 2:
        raise SystemExit(
            '사용법: python gemini_vision.py "images/products.jpg"'
        )

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        raise SystemExit(f"사진 파일을 찾을 수 없습니다: {image_path}")

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }
    mime_type = mime_types.get(image_path.suffix.lower())

    if mime_type is None:
        raise SystemExit("JPG 또는 PNG 사진을 사용하세요.")

    image_part = types.Part.from_bytes(
        data=image_path.read_bytes(),
        mime_type=mime_type,
    )

    prompt = """
사진 속 작업대에 놓인 상품을 관찰하고 한국어로 답하세요.

1. 구분 가능한 상품마다 번호를 붙이세요.
2. 각 상품의 종류, 주요 색상, 이미지에서의 위치를 설명하세요.
   위치는 왼쪽/가운데/오른쪽, 위쪽/아래쪽으로 표현하세요.
3. '파란색 상품을 가져다줘'라는 요청에 해당하는 상품을 고르세요.
4. 대상이 여러 개라면 추가 질문이 필요하다고 답하세요.
5. 대상이 없으면 없다고 답하세요.
6. 불확실한 브랜드나 캐릭터 이름은 추측하지 마세요.

이 단계에서는 로봇을 움직이지 않고 관찰 결과만 설명하세요.
"""

    print(f"사진 전송 중: {image_path.name}")

    with genai.Client(api_key=api_key) as client:
        response = client.models.generate_content(
            model="gemini-robotics-er-2-preview",
            contents=[image_part, prompt],
        )

    if not response.text:
        raise SystemExit("텍스트 응답이 없습니다. 응답 상태를 확인하세요.")

    print("\n인식 결과:")
    print(response.text)

    output_path = Path("vision_result.txt")
    output_path.write_text(response.text, encoding="utf-8")
    print(f"\n결과 저장 완료: {output_path.resolve()}")


if __name__ == "__main__":
    main()