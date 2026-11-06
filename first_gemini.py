# gemini api 연결을 테스트하는 간단한 예제
import os

from google import genai


def main():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise SystemExit("GEMINI_API_KEY를 먼저 설정하세요.")

    print("Gemini에 요청을 보내는 중입니다...")

    with genai.Client(api_key=api_key) as client:
        response = client.models.generate_content(
            model="gemini-robotics-er-2-preview",
            contents=(
                "한국어로 짧게 답하세요. "
                "사용자가 '파란색 상품을 가져다줘'라고 요청했습니다. "
                "로봇이 실행 전에 확인해야 할 사항 세 가지를 알려주세요."
            ),
        )

        if response.text:
            print("\n응답:")
            print(response.text)
        else:
            print("응답은 도착했지만 텍스트가 없습니다.")
            print(response)


if __name__ == "__main__":
    main()