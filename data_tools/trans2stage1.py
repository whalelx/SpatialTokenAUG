import json



src_name = "/data/spatialRGPT_qa/cot/train_20250421.json"
save_name = "/data/spatialRGPT_qa/cot/train_20250421_stage1.json"

new_data = []
with open(src_name, "r") as r_op:
    data = json.load(r_op)

    for dialogue in data:
        # print(dialogue)
        # question, answer, cot, image_path


        template = {
            "filename": dialogue["image_path"],
            "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + "The question related to spatial reasoning is: " + dialogue["question"]
            },
            {
                "from": "gpt",
                "value": dialogue["cot"]
            }]

        }

        new_data.append(template)


with open(save_name, "w") as w_op:
    json.dump(new_data, w_op, ensure_ascii=False, indent=4)
