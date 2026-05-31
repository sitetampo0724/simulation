import time
import requests
import json
import base64
import sys

class photograph:


    image_path = None


    def __init__(self , pathimage):
        self.image_path=pathimage


    def recognize_image_flexible(self,custom_prompt=None):
        """
        灵活的图片识别，让AI自己组织带emoji的回复
        """
        url = "https://models.sjtu.edu.cn/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-rSDGyjxTopaYoI_d16iHjw"
        }

        with open(self.image_path, 'rb') as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # 默认prompt
        if custom_prompt is None:
            user_prompt = """请描述这张图片，要求：
    1. 使用Markdown格式（标题、列表、加粗等）
    2. 加入合适的emoji表情（如🎨📷🌟👤🏠等）
    3. 结构清晰，有层次感
    4. 语言生动有趣

    请开始你的描述："""
        else:
            user_prompt = custom_prompt

        messages = [
            {
                "role": "system",
                "content": "你是一个擅长用emoji和Markdown格式回答的AI助手，回答要生动有趣、格式优美。"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

        data = {
            "messages": messages,
            "stream": True,  # 这个已经是True，保持流式输出
            "temperature": 0.8,
            "model": "qwen3vl",
        }

        try:
            response = requests.post(url, headers=headers, json=data, stream=True, timeout=120)

            if response.status_code == 200:
                print("\n✨ AI 分析中...\n")

                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data: "):
                            line_str = line_str[6:]

                        if line_str == "[DONE]":
                            print("\n\n✅ 完成")
                            break

                        try:
                            chunk = json.loads(line_str)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    # 逐字输出，每个字之间稍微停顿
                                    for char in content:
                                        print(char, end="", flush=True)
                                        time.sleep(0)  # 控制输出速度，可以调整这个数值
                        except json.JSONDecodeError:
                            continue

            else:
                print(f"❌ 请求失败: {response.status_code}")

        except Exception as e:
            print(f"⚠️ 错误: {str(e)}")