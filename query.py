import time
import requests
import json
import sys

class query:

    def __init__(self):
        pass

    def query_stream(self,messages, use_emoji=True, style="creative"):
        """
        流式对话函数，支持emoji和Markdown格式输出

        参数:
            messages: 用户输入的消息
            use_emoji: 是否使用emoji格式
            style: 回复风格 - "creative"(创意), "professional"(专业), "simple"(简洁)
        """
        url = "https://models.sjtu.edu.cn/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-rSDGyjxTopaYoI_d16iHjw"
        }

        # 根据风格设置不同的system prompt
        style_prompts = {
            "creative": """你是一个富有创意的AI助手，擅长使用emoji和Markdown格式让回答生动有趣。
    要求：
    1. 适当使用emoji表情（如🎨📝💡✨🎯📚💻等）
    2. 使用Markdown格式（标题、列表、加粗、引用等）
    3. 回答要有层次感，结构清晰
    4. 语言生动自然，有亲和力""",

            "professional": """你是一个专业的AI助手，回答要专业、准确。
    要求：
    1. 使用适当的emoji作为视觉标记（如📊✅⚠️💡等）
    2. 使用Markdown格式组织内容
    3. 逻辑清晰，重点突出
    4. 保持专业但友好的语气""",

            "simple": """你是一个简洁的AI助手，回答要简短明了。
    要求：
    1. 使用简单emoji标记重点
    2. 可以用Markdown列表
    3. 回答控制在3-5句话以内"""
        }

        # 构建消息列表
        chat_messages = []

        # 添加system prompt（如果需要emoji格式）
        if use_emoji:
            system_prompt = style_prompts.get(style, style_prompts["creative"])
            chat_messages.append({"role": "system", "content": system_prompt})

        # 添加用户消息
        chat_messages.append({"role": "user", "content": f"{messages}"})

        print("💬 正在生成回复...\n")
        print("=" * 60)

        data = {
            "messages": chat_messages,
            "stream": True,
            "do_sample": True,
            "repetition_penalty": 1.00,
            "temperature": 0.8 if use_emoji else 1e-5,  # 使用emoji时提高温度
            "top_k": 20,
            "model": "deepseek-chat",
        }

        try:
            response = requests.post(url, headers=headers, json=data, stream=True, timeout=30)

            if response.status_code == 200:
                full_content = ""
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            line = line[6:]
                            if line.strip() == '[DONE]':
                                break
                            try:
                                chunk = json.loads(line)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        full_content += content
                                        # 逐字输出
                                        for char in content:
                                            print(char, end='', flush=True)
                                            time.sleep(0)  # 控制输出速度
                            except json.JSONDecodeError:
                                continue
                print("\n" + "=" * 60)
                print("✅ 回复完成\n")
                return full_content
            else:
                error_msg = f"❌ 请求失败: {response.status_code}"
                print(error_msg)
                return error_msg
        except Exception as e:
            error_msg = f"⚠️ 请求出错: {str(e)}"
            print(error_msg)
            return error_msg

