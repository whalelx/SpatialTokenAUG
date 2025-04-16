from openai import OpenAI
import os
import base64
import jsonlines
import json


sys_prompt1 = """## Role: Visual-Spatial Reasoning Expert.
## Task: 
- You will be provided with an image and a dialogue. There are one to two targets in the image outlined with light blue boxes, and each box is labeled with a number in the top-left corner.
- Based on the given image, spatial-reasoning related question and answer, extract other objects in the image that help with the question in dialogue, excluding the outlined targets. 
- Answer the question with "Yes" or "No". If the answer is "Yes", list all other objects that help with the question in dialogue one by one, and tell me how they help with the quesiton.
- Do not output the abstract objectes such as "light", background" which cannot be framed within the detection bounding boxes.

## Output Example: (Must be in JSON format)
{
    "exist": "Whether other objects relevant to the dialogue exist",
    "desc": "Other objects relevant to the dialogue"
}
"""

sys_prompt = "The targets in region are outlined in light blue boxes and each box is labeled with a number in the top-left corner. \
    Describe the object in Region [0] and Region [1]. \
    Identify the most fine-grained but whold object in the bounding box.\
    for example: if an individual is riding a motorcycle and only the motorcycle is bounded, the object should be the motorcycle rather than the person\
        "

 
# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
 

def run_single_image_idealab(question, image_path):
    base64_image = encode_image(image_path)
    client = OpenAI()
     
    messages=[
        {
            "role": "system", 
             "content": sys_prompt,
        },
        {
            "role": "user", 
             "content": [
                {"type":"text", "text": question},
                {
                   "type":"image_url",
                   "image_url":{
                      "url":f"data:image/png;base64,{base64_image}",
                      }
                }
            ]
        }
    ]
    completion = client.chat.completions.create(
          model="gpt-4o-0513-global",
          messages=messages
    )
    chat_response = completion
    answer = chat_response.choices[0].message.content
    return answer




if __name__ == "__main__":
    # answer = run_single_image_idealab(question, "0.png")
    # print(answer)

    image_dir = "/data/spatialRGPT_test3/images"

    with open("/data/spatialRGPT_test3/jsons/00fe670dda70e4b4.json") as r_op:
        data = json.load(r_op)
        for datum in data:
            qu = datum["question"]
            an = datum["answer"]
            image_path = os.path.join(image_dir, datum["image_path"])
            
            question = str({"question": qu, "answer": an})
            # question = str({"question": qu})
            print()
            print(image_path)
            print(question)
            answer = run_single_image_idealab(question, image_path)
            print(answer)
             

