import json
import os
from collections import defaultdict

def convert_to_conversations(data):
    # 以 image_id 为键，聚合问答
    grouped = defaultdict(list)
    
    for item in data:
        image_path = item["image_path"]
        # 提取唯一 image ID，例如：spatialRGPT_qa/images/00000000/**460ebc433f62ab48**/0.png
        parts = image_path.split("/")
        if len(parts) >= 4:
            image_id = parts[3]
        else:
            continue
        grouped[image_id].append({
            "question": item["question"],
            "answer": item["answer"]
        })

    conversations_data = []

    for image_id, qas in grouped.items():
        conversation = []
        for i, qa in enumerate(qas):
            question = qa["question"]
            if i == 0:
                question = "<image>\n" + question  # 仅在第一轮加 image token
            
            conversation.append({
                "from": "human",
                "value": question
            })
            conversation.append({
                "from": "gpt",
                "value": qa["answer"]
            })
        conversations_data.append({
            "filename": image_id,
            "conversations": conversation
        })

    return conversations_data


# 示例：加载 JSON 文件
with open("your_input_file.json", "r", encoding="utf-8") as f:
    input_data = json.load(f)

# 转换
converted = convert_to_conversations(input_data)

# 输出为新的 JSON 文件
with open("converted_conversations.json", "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=4, ensure_ascii=False)
