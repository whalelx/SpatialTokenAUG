import json
import os
import shutil
import tqdm


json_path = "/data/dataset-spatial-reasoning/spatialrgpt/result_10_depth_convs.json"
image_dir = "/data/dataset-spatial-reasoning/openimagev7/train/"


save_dir = "/data/spatialRGPT_split2"
os.makedirs(save_dir, exist_ok=True)


with open(json_path, "r") as r_op:
    list_data_dict = json.load(r_op)
    
    chunk_size = 1000
    chunk_num = (len(list_data_dict) + chunk_size - 1) // chunk_size

    # for chunk_start in range(0, len(list_data_dict), chunk_size):
    for chunk_idx in tqdm.tqdm(range(chunk_num)):
        chunk_start = chunk_idx * chunk_size
        chunk_end = (chunk_idx + 1) * chunk_size

        save_json_path = os.path.join(save_dir, str(chunk_idx).zfill(8) + ".json")
        with open(save_json_path, "w", encoding='utf-8') as w_op:
            data = []
            
            for datum in list_data_dict[chunk_start:chunk_end]:
                del datum["rle"]
                data.append(datum)
            
            json.dump(data, w_op, ensure_ascii=False, indent=4)

