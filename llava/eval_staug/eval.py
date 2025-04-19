# add by @lx
import argparse
import copy
import json
import math
import os
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as cocomask
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

def pad_to_square(array):
    H, W = array.shape
    max_side = max(H, W)

    padded_array = np.zeros((max_side, max_side), dtype=np.uint8)
    pad_h = (max_side - H) // 2
    pad_w = (max_side - W) // 2
    padded_array[pad_h : pad_h + H, pad_w : pad_w + W] = array

    return padded_array


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def clamp_box(bbox, image_info):
    h, w = image_info["height"], image_info["width"]
    bbox[0] = max(min(w, bbox[0]), 0)
    bbox[2] = max(min(w, bbox[2]), 0)

    bbox[1] = max(min(h, bbox[1]), 0)
    bbox[3] = max(min(h, bbox[3]), 0)

Action_tokens = {
    "region_x": "<|x_0|>,<|x_1|>,<|x_2|>,<|x_3|>,<|x_4|>,<|x_5|>,<|x_6|>,<|x_7|>".split(","),
    "region_y": "<|y_0|>,<|y_1|>,<|y_2|>,<|y_3|>,<|y_4|>,<|y_5|>,<|y_6|>,<|y_7|>".split(","),
}


def check_region_tokens(text):

    pattern = re.compile(r'<\|region_token_start\|>(<\|[xy]_[01234567]\|>)+<\|region_token_end\|>')
    matches = pattern.finditer(text)

    found_tokens = []
    for match in matches:
        match_str = match.group()
        # match_str = match_str.replace("<|region_token_start|>","").replace("<|region_token_end|>","")
        match_tokens_x = [token for token in Action_tokens["region_x"] if token in match_str]
        match_tokens_y = [token for token in Action_tokens["region_y"] if token in match_str]
        found_tokens.append((match_tokens_x, match_tokens_y))

    if found_tokens:
        return found_tokens, True
    else:
        return None, False

def region_token_to_new_image_path(found_region_tokens, meta_img_path, num_cut=8):
    regions = from_region_tokens_to_region(found_region_tokens)
    max_x = max([x for x,y in regions])
    min_x = min([x for x,y in regions])
    max_y = max([y for x,y in regions])
    min_y = min([y for x,y in regions])

    max_x = min(max_x+1, num_cut-1)
    min_x = max(min_x-1, 0)
    max_y = min(max_y+1, num_cut-1)
    min_y = max(min_y-1, 0)

    image_path = meta_img_path
    image = Image.open(image_path)
    width, height = image.size
    st_w = 0
    st_h = 0

    grid_width = width / num_cut
    grid_height = height / num_cut

    new_x_min = st_w + (min_x ) * grid_width
    new_y_min = st_h + (min_y ) * grid_height
    new_x_max = st_w + (max_x + 1) * grid_width
    new_y_max = st_h + (max_y + 1) * grid_height

    new_bbox = (new_x_min, new_y_min, new_x_max, new_y_max)
    bbox_subfix = f"_{int(new_bbox[0])},{int(new_bbox[1])},{int(new_bbox[2])},{int(new_bbox[3])}"

    new_image_path = image_path + bbox_subfix
    return new_image_path


def from_region_tokens_to_region(region_tokens):
    match_tokens_x, match_tokens_y = region_tokens
    x_indices = [int(item.replace("<|x_","").replace("|>","")) for item in match_tokens_x]
    y_indices = [int(item.replace("<|y_","").replace("|>","")) for item in match_tokens_y]
    minx = min(x_indices)
    miny = min(y_indices)
    maxx = max(x_indices)
    maxy = max(y_indices)
    region = [(minx, miny), (maxx, maxy)]
    return region

def generate_region_item(meta_img_path, generated_answer, index, ls_found_region_tokens, num_cut=8):

    ls_new_image_path = [region_token_to_new_image_path(found_region_tokens, meta_img_path, num_cut=8) for found_region_tokens in ls_found_region_tokens]
    
    if len(ls_new_image_path) == 1:
        new_image_prompt = "<image>"
    else:
        new_image_prompt = ""
        for i in range(len(ls_new_image_path)):
            new_image_prompt += f"Related region {i}: <image>\n"

    return new_image_prompt, ls_new_image_path

class Evaluator:
    def __init__(self, args):
        self.args = args

        # Model
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_name, args.model_base)
        self.model.to(dtype=torch.bfloat16)
        self.stop_str = (
            conv_templates[args.conv_mode].sep
            if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO
            else conv_templates[args.conv_mode].sep2
        )
    # NOTE: only need to change this function when changing dataset
    def eval_to_files(self):
        with open(args.annotation_file) as f:
            questions = json.load(f)
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
        answers_file = os.path.expanduser(args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")

        for line in tqdm(questions, total=len(questions)):
            question_id = line["id"]
            text_question = line["text_q"]
            qa_info = line["qa_info"]

            image_files = line["image_info"]["file_path"]
            conversations = line["conversations"]

            output_list = self.eval_model(image_files, conversations)
            
            answer = [output.update({
                "qa_info": qa_info,
                "question_id": question_id,
                "image": image_files,
                "question": text_question,
            }) for output in output_list]


            ans_file.write(
                json.dumps(answer)
                + "\n"
            )
        ans_file.close()

    def eval_model(self, image_files, conversations):
        args = self.args
        tokenizer, model, image_processor, context_len = self.tokenizer, self.model, self.image_processor, self.context_len

        image_files = [os.path.join(args.image_folder, image_file) for image_file in image_files]
        images_tensor = process_images(image_files, image_processor, model.config).to(model.device, dtype=torch.float16)

        num_sub_question = len(conversations) // 2
        return_list = []
        for i in range(num_sub_question):
            question = conversations[i * 2]["value"]
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], question)
            outputs = None

            for i in range(2): # run two rounds
                if outputs is not None:
                    geneated_answer = outputs
                    ls_found_region_tokens, do_region = check_region_tokens(geneated_answer)
                    if do_region:
                        new_image_prompt, ls_new_image_path = generate_region_item(image_files[0], geneated_answer, i, ls_found_region_tokens)
                        ls_new_image_path = [os.path.join(args.image_folder, image_file) for image_file in ls_new_image_path]
                        images_tensor_aug = process_images(ls_new_image_path, image_processor, model.config).to(model.device, dtype=torch.float16)
                        image_files += ls_new_image_path
                        images_tensor = torch.cat([images_tensor, images_tensor_aug], axis=0)
                        conv.append_message(conv.roles[0], new_image_prompt)
                    else: 
                        break # Do not generate again when no clip region is required

                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                input_ids = input_ids.to(device="cuda", non_blocking=True)
                input_ids = input_ids.unsqueeze(0)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=images_tensor.to(dtype=torch.bfloat16, device="cuda", non_blocking=True),
                        depths=None,
                        masks=None,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=128,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id, # TODO @lx
                    )

                outputs = outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(self.stop_str):
                    outputs = outputs[: -len(self.stop_str)]
                outputs = outputs.strip()

                conv.append_message(conv.roles[1], outputs)


            return_list.append({
                "pred": outputs,
                "gt": conversations[i * 2 + 1]["value"],
            })
        
        return return_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--annotation-file", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--use-mask", type=bool, default=True)
    args = parser.parse_args()

    disable_torch_init()

    eval_model(args)