import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5"

import time
from PIL import Image
from transformers import pipeline

# ========== box 处理函数 ==========
import re
def add_box_token(input_string):
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            coordinates = re.findall(r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)
            updated_action = action
            for coord_type, x, y in coordinates:
                updated_action = updated_action.replace(
                    f"{coord_type}='({x},{y})'",
                    f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'"
                )
            processed_actions.append(updated_action)
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string

# ========== pipeline 加载 ==========
# 使用本地路径加载模型
# 请确保你的脚本和模型目录（GUI-Actor-2B-Qwen2-VL）在同一个目录下
local_model_path = "./GUI-Actor-2B-Qwen2-VL"
pipe = pipeline("image-text-to-text", model=local_model_path)

# ========== 初始化 messages ==========
messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "你是一个能够操作计算机界面的智能体。你会观察截图，理解用户需求，然后输出 Thought 和 Action。"}
        ]
    }
]

# ========== 多轮对话函数 ==========
def run_agent(messages, screenshot_path, user_text):
    image = Image.open(screenshot_path)

    # 新一轮用户输入
    user_message = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text}
        ]
    }
    messages.append(user_message)

    # 调用 pipeline
    result = pipe(text=messages)
    output_text = result[0]["generated_text"]
    print("模型输出：\n", output_text)

    # 保存 assistant 响应
    messages.append({
        "role": "assistant",
        "content": [
            {"type": "text", "text": add_box_token(output_text)}
        ]
    })

    return messages

# ========== 示例运行 ==========
if __name__ == "__main__":
    for _ in range(10):
        start_time = time.time()
        messages = run_agent(messages, "./screenshot.png", "点击搜索框并输入 'hello'")
        end_time = time.time()
        print("Latency:", end_time - start_time, "seconds")

    messages = run_agent(messages, "./screenshot.png", "然后按回车键")

    print("\n=== 最终 messages ===")
    for m in messages:
        print(m["role"], ":", m["content"])