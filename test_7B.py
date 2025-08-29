import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5"
import re
import time
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# ========== box 处理函数 ==========
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


# ========== 模型加载 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "ByteDance-Seed/UI-TARS-1.5-7B"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

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
def run_agent(messages, screenshot_path, user_text, max_new_tokens=200):
    # 添加用户输入
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "path": screenshot_path},
            {"type": "text", "text": user_text}
        ]
    })

    # 构造输入
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # 生成输出
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    print("模型输出：\n", decoded)

    # 保存到 messages（作为 assistant 回复）
    messages.append({
        "role": "assistant",
        "content": [
            {"type": "text", "text": add_box_token(decoded)}
        ]
    })

    return messages


# ========== 示例运行 ==========
if __name__ == "__main__":
    # 第 1 轮
    start_time = time.time()
    messages = run_agent(messages, "./screenshot.png", "点击搜索框并输入 'hello'")
    end_time = time.time()
    print("Latency:", end_time - start_time, "seconds")
    
    # 第 2 轮（继续带上下文）
    messages = run_agent(messages, "./screenshot.png", "然后按回车键")

    # 打印完整对话
    print("\n=== 最终 messages ===")
    for m in messages:
        print(m["role"], ":", m["content"])
