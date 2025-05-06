import jsonlines
import json
import os
import tqdm
from openai import OpenAI

sys_prompt = ''' ##角色: 专业的翻译人员。
## 任务: 将中文翻译为英文。
- 请直接输出翻译后的英文，不包含任何引号。
- "区域0"等术语请翻译为"Region [0]"
'''



def translate_idealab(question):
    client = OpenAI()
     
    messages=[
        {
            "role": "system", 
             "content": sys_prompt,
        },
        {
            "role": "user", 
             "content": "请将下列中文翻译为英文: " + question,
        }
    ]
    completion = client.chat.completions.create(
          model="gpt-4o-0513-global",
          messages=messages
    )
    chat_response = completion
    answer = chat_response.choices[0].message.content
    return answer



def idx2token(x1,y1,x2,y2, token_len=8):
    output_string = "<region_token_start>"
    output_string += "<x_" + str(int(x1 * token_len)) + ">"
    output_string += "<y_" + str(int(y1 * token_len)) + ">"
    output_string += "<x_" + str(int(x2 * token_len)) + ">"
    output_string += "<y_" + str(int(y2 * token_len)) + ">"
    output_string += "<region_token_end>"
    return output_string




src_dir = "/data/spatialRGPT_qa/itags/spatial_anno0428/2"
src_data_path = "/data"
save_name = "/data/spatialRGPT_qa/train_20250428_2.json"


bad_list = []
unlabeled_list = []


dialogue_list = []
for file_name in tqdm.tqdm(sorted(os.listdir(src_dir))):
    print(file_name)
    file_path = os.path.join(src_dir, file_name)

    with jsonlines.open(file_path, "r") as r_op:
        for item in r_op:
            # print(item.keys())
            # ['任务ID', '子任务包ID', '数据集ID', '数据ID', 'image', 'dialogue', '框选-PICTURE-undefined', '请对数据质量进行评估', '若回答错误，请修正(英文)', '子任务包状态', '最终更新时间', '是否废弃', '废弃原因', '标注环节结果', '标注环节人员', '检查环节结果', '检查环节人员', '检查环节是否标记错误', '检查环节是否修改答案', '验收环节结果', '验收环节人员', '验收环节是否标记错误', '验收环节是否修改答案', '子任务包地址']

            if item["请对数据质量进行评估"] is None:
                # 尚未标注
                unlabeled_list.append(item)
                continue

            if isinstance(item["请对数据质量进行评估"], str):
                if ("图片存在问题" == item["请对数据质量进行评估"] or
                    "问答正确性难以判断" == item["请对数据质量进行评估"] or
                    "其他问题" == item["请对数据质量进行评估"]):
                    # 质量不合格
                    continue
                elif "问答错误" == item["请对数据质量进行评估"]:
                    dialogue = item["若回答错误，请修正(英文)"]
                    assert(dialogue is not None)
                elif "完全正确" == item["请对数据质量进行评估"]:
                    dialogue = item["dialogue"]
                else:
                    bad_list.append(item)

            elif isinstance(item["请对数据质量进行评估"], list):
                if ("图片存在问题" in item["请对数据质量进行评估"] or
                    "问答正确性难以判断" in item["请对数据质量进行评估"] or
                    "其他问题" in item["请对数据质量进行评估"]):
                    # 质量不合格
                    continue
                elif "问答错误" in item["请对数据质量进行评估"]:
                    dialogue = item["若回答错误，请修正(英文)"]
                    assert(dialogue is not None)
                elif "完全正确" in item["请对数据质量进行评估"]:
                    dialogue = item["dialogue"]
                else:
                    bad_list.append(item)

            else:
                raise ValueError("Value Error!") 


            # {"orientation":0,"objects":[{"polygon":{"ptList":[{"x":38.84626517397822,"y":100.47260938116084},{"x":273.9949519595692,"y":100.47260938116084},{"x":273.9949519595692,"y":306.762150071472},{"x":38.84626517397822,"y":306.762150071472}]},"name":"正常男性的手臂长度为70-80厘米之间","type":1,"id":"c97a476f-12bd-445c-92e6-556da5b5ed67","color":"#9254DE","result":{"请对该目标进行简要描述(中文)":"正常男性的手臂长度为70-80厘米之间","该目标和问答相关的原因是(中文)":"题目提到它们之间相距 1.42 英寸，很明显是错误的"},"lineStyle":"solid"}],"width":683,"height":1024}


            cot_list = []

            # msg retr 
            image_name = item["image"]
            tmp_list = image_name.split("/")[-5:-1]
            tmp_list[-1] = tmp_list[-1] + ".json"
            tmp_list[1] = "jsons"
            json_path = os.path.join(src_data_path, "/".join(tmp_list))

            json_idx = int(image_name.split("/")[-1].split(".")[0])
            with open(json_path, "r") as r_op2:
                data = json.load(r_op2)
                json_msg = data[json_idx]
          
            print(json_path)
            print(json_msg)
            raw_mask_list = json_msg["mask_list"]    # x1, y1, x2, y2
            raw_region_list = json_msg["region_list"]
            raw_image_width = json_msg["image_width"] 
            raw_image_height = json_msg["image_height"]

             
            for raw_mask, raw_region_id in zip(raw_mask_list, raw_region_list):
                raw_mask_token = idx2token(raw_mask[0] / raw_image_width, 
                                       raw_mask[1] / raw_image_height, 
                                       raw_mask[2] / raw_image_width, 
                                       raw_mask[3] / raw_image_height)
                raw_region_description = f"Region [{raw_region_id}]"
                # print(raw_mask_token)
                # print(raw_region_description)

                cot_list.append({"mask_token": raw_mask_token, "region_description": raw_region_description})
                

            box_msg = item["框选-PICTURE-undefined"]
            if box_msg is not None:
                box_msg = eval(box_msg)
                image_width = box_msg["width"]
                image_height = box_msg["height"]
                assert(image_width == raw_image_width)
                assert(image_height == raw_image_height)

                objects = box_msg["objects"]
                
                bad_flag = False
                for obj in objects:
                    box = obj["polygon"]["ptList"]
                    if len(box) != 4:
                        bad_flag = True
                        break
                if bad_flag:
                    bad_list.append(item)
                    continue

                for obj in objects:
                    box = obj["polygon"]["ptList"]
                    result = obj["result"]
                    description = result["请对该目标进行简要描述(中文)"]
                    reason = result["该目标和问答相关的原因是(中文)"]
                    
                    description = translate_idealab(description)
                    reason = translate_idealab(reason)

                    # x is width, y is height

                    min_x = box[0]["x"] / image_width
                    min_y = box[0]["y"] / image_height
                    max_x = box[1]["x"] / image_width
                    max_y = box[2]["y"] / image_height

                    # print([min_x / image_width, max_x / image_width, min_y / image_height, max_y / image_height])
                    mask_token = idx2token(min_x, min_y, max_x, max_y)
                    # region_description = "Description: " + description + ". Reason: " + reason
                    # print(mask_token)
                    # print(region_description)
                    
                    cot_list.append({"mask_token": mask_token, "region_description": description, "reason": reason})
                        

            cot_string = ""
            for cot in cot_list:
                tmp_string = cot["region_description"] + cot["mask_token"] + ". "
                if "reason" in cot:
                    tmp_string += " " + cot["reason"]
                cot_string += tmp_string

            try:
                dialogue = eval(dialogue)
            except:
                continue
            dialogue["cot"] = cot_string
            # https://city-brain-vendor.oss-cn-hangzhou.aliyuncs.com/guoxin/Spatial/datasets/spatialRGPT_qa/images/00000000/3f94d1e5c358df2a/4.png
            save_image_name = "/".join(image_name.split("/")[-5:])
            dialogue["image_path"] = save_image_name
            
            '''
            template = {
                "filename": save_image_name,
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
            '''
            dialogue_list.append(dialogue)


with open(save_name, "w") as w_op:
    json.dump(dialogue_list, w_op, ensure_ascii=False, indent=4)

print("bad_list: ", bad_list)
print("unlabeled_list: ", unlabeled_list)



