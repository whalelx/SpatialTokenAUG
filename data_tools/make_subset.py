import json
import os
import shutil


json_path = "/data/dataset-spatial-reasoning/spatialrgpt/result_10_depth_convs.json"
image_dir = "/data/dataset-spatial-reasoning/openimagev7/train/"


save_dir = "/data/spatialRGPT_test"
save_json_path = os.path.join(save_dir, "train.json") 
save_image_dir = os.path.join(save_dir, "train")
os.makedirs(save_image_dir, exist_ok=True)


with open(save_json_path, "w", encoding='utf-8') as w_op:
    data = []
    with open(json_path, "r") as r_op:
        list_data_dict = json.load(r_op)
        

        # data = list_data_dict[:100]
        for datum in list_data_dict:
            image_name = datum["filename"]
            image_path = os.path.join(image_dir, image_name + ".jpg")
            save_image_path = os.path.join(save_image_dir, image_name + ".jpg")

            try:
                print()
                print(image_path)
                print(save_image_path)
                shutil.copyfile(image_path, save_image_path)
                data.append(datum)
            except:
                pass
            
            if len(data) == 100:
                break

    json.dump(data, w_op, ensure_ascii=False, indent=4)

