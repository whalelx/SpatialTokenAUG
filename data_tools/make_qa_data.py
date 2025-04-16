import cv2
import copy
import os
import json
import re
import tqdm
import numpy as np


def replace_with_list(s, replacements):
    counter = {'index': 0}

    def replacement_function(match):
        replacement = replacements[counter['index']]
        counter['index'] += 1
        return replacement

    result = re.sub(r'<mask> <depth>', replacement_function, s)
    return result


def clamp(bbox, image_h, image_w):
    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, min(x1, image_w))
    x2 = max(0, min(x2, image_w))

    y1 = max(0, min(y1, image_h))
    y2 = max(0, min(y2, image_h))

    return x1, y1, x2, y2


def process_msg(conversations, image_np, mask_np, bbox, image_name, save_dir):
    mask_count = 0
    conv_count = len(conversations) // 2
    output_list = []
    for idx in range(conv_count):
        q_idx = 2 * idx
        a_idx = 2 * idx + 1

        assert(conversations[q_idx]["from"] == "human")
        assert(conversations[a_idx]["from"] == "gpt")

        question = conversations[q_idx]["value"]
        answer = conversations[a_idx]["value"]

        tmp_count = question.count("<mask> <depth>")

        label_list = [i for i in range(tmp_count)] 
        mask_idx_list = [mask_count + i for i in range(tmp_count)]
        mask_count += tmp_count

        
        tmp_list = []
        for tmp_idx, mask_idx in enumerate(mask_idx_list):
            if bbox:
                element = bbox[mask_idx]
            if mask_np:
                element = mask_np[mask_idx]

            if element not in tmp_list:
                tmp_list.append(element)
            else:
                tmp_idx2 = tmp_list.index(element)
                label_list[tmp_idx] = label_list[tmp_idx2]

        replacements = [f"Region [{i}]" for i in label_list]
        new_question = replace_with_list(question, replacements)
        new_question = re.sub(r"<image>\n", "", new_question)


        masked_image = copy.deepcopy(image_np)
        for label, mask_idx in zip(label_list, mask_idx_list):
            if mask_np:
                mask_ = mask_np[mask_idx]
                mask_new = np.zeros_like(masked_image)
                mask_new[:,:,0] = mask_ * 255

                alpha = 1 # toumingdu
                masked_image = cv2.addWeighted(masked_image, 1.0, mask_new, alpha, 0)
                masked_image = np.where((mask_new.sum(-1)>0).any(), mask_new, masked_image)
            
            if bbox:
                box = bbox[mask_idx]
                x1, y1, x2, y2 = clamp(box, image_info["height"], image_info["width"])
                icon_x, icon_y, icon_w, icon_h = x1+5, y1+5, 20, 20
                cv2.rectangle(masked_image, (icon_x, icon_y), (icon_x + icon_w, icon_y + icon_h), (0,0,0), -1)
                cv2.rectangle(masked_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

                label = str(label)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                text_color = (255,255,255)

                label_height = icon_h
                text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                text_x = icon_x + (icon_w - text_size[0]) // 2
                text_y = icon_y + label_height - (label_height - text_size[1]) // 2

                cv2.putText(masked_image, label, (text_x, text_y), font, font_scale, text_color, font_thickness)

        save_image_dir = os.path.join(save_dir, image_name)
        os.makedirs(save_image_dir, exist_ok=True)
        save_image_path = os.path.join(save_image_dir , str(idx) +  ".png")
        cv2.imwrite(save_image_path, masked_image)

        json_image_path = os.path.join(image_name, str(idx) +  ".png")

        msg = {"question": new_question, "answer": answer, "image_path": json_image_path}
        output_list.append(msg)

    return output_list


def process_mask(rles, bboxes, image_info, modality="mask"):
    masks = []
    if modality == "rle":
        for rle in rles:
            m = cocomask.decode(rle)
            m = m.astype(np.uint8)
            masks.append(m)

    elif modality == "mask":
        for bbox in bboxes:
            zero_mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
            x1, y1, x2, y2 = clamp(bbox, image_info["height"], image_info["width"])
            # zero_mask[y1:y2, x1:x2] = 1
            # Draw only the outline of the bounding box
            # Top and bottom horizontal lines
            if y2 > y1:
                zero_mask[y1:y1+5, x1:x2+1] = 1
                zero_mask[y2:y2+5, x1:x2+1] = 1

            # Left and right vertical lines
            if x2 > x1:
                zero_mask[y1:y2, x1:x1+5] = 1
                zero_mask[y1:y2, x2:x2+5] = 1

            masks.append(zero_mask)

    return np.array(masks)




data_path = "/data/spatialRGPT_test/train.json"
image_folder = "/data/spatialRGPT_test/train"

save_json = "/data/spatialRGPT_test3/jsons"
save_dir = "/data/spatialRGPT_test3/images"
os.makedirs(save_json, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)


with open(data_path) as r_op:
    data = json.load(r_op)

for item in tqdm.tqdm(data):
    filename = item["filename"]
    
    with open(os.path.join(save_json, filename + ".json"), "w") as w_op:
        print(filename)
        conversations = item["conversations"]
        rle = item["rle"]
        bbox = item["bbox"]

        image_path = os.path.join(image_folder, filename + ".jpg")
        raw_image = cv2.imread(image_path)

        height, width = raw_image.shape[:2]
        image_info = {"height": height, "width": width}

        # masks = process_mask(rle, bbox, image_info)
        output_list = process_msg(conversations, raw_image, None, bbox, filename, save_dir)

        json.dump(output_list, w_op, indent=4)

