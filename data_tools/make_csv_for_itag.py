import os
import csv
import json
import tqdm
import oss2
import random


def get_image_list(image_dir, update=False):
    # image_dir = "guoxin/tmp/巡店项目/" # 注意：最后的斜杠必须要带上

    # 初始化客户端
    # 从环境变量中获取访问凭证。运行本代码示例之前，请确保已设置环境变量OSS_ACCESS_KEY_ID和OSS_ACCESS_KEY_SECRET。
    # auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
    auth = oss2.Auth('xxx', 'xxx')


    # 填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
    endpoint = "oss-cn-hangzhou.aliyuncs.com"

    # 填写Endpoint对应的Region信息，例如cn-hangzhou。注意，v4签名下，必须填写该参数
    # region = "cn-hangzhou"

    # yourBucketName填写存储空间名称。
    # bucket = oss2.Bucket(auth, endpoint, "", region=region)
    bucket = oss2.Bucket(auth, endpoint, 'city-brain-vendor')


    # 列举fun文件夹下的文件与子文件夹名称，不列举子文件夹下的文件。
    # for obj in oss2.ObjectIterator(bucket, prefix='guoxin/tmp/巡店项目/', delimiter ='/'):


    file_list = []
    # 列举所有文件和文件夹
    for obj in oss2.ObjectIterator(bucket, prefix=image_dir):
        # 通过is_prefix方法判断obj是否为文件夹。（请注意判断是否为文件夹需要配置delimiter和prefix来完成模拟文件夹功能）

        if obj.key.endswith('/'):  # 如果是目录，则修改目录权限
            print('directory: ' + obj.key)
            if update:
                bucket.put_object_acl(obj.key, oss2.OBJECT_ACL_PUBLIC_READ)
        else:  # 如果是文件，则修改文件权限
            print('file: ' + obj.key)
            if update:
                bucket.update_object_meta(obj.key, {'x-oss-object-acl': oss2.OBJECT_ACL_PUBLIC_READ})

            # print(obj.key, obj.etag, obj.last_modified, obj.storage_class)
            # 构建访问URL（注意：这里仅为示例，实际使用时请根据实际情况调整）
            url = f"https://{bucket.bucket_name}.{endpoint}/{obj.key}"
            file_list.append(url)

    return file_list



def run(chunk_name):
    json_dir = "/data/spatialRGPT_qa/jsons/" + chunk_name

    csv_dir = "/data/spatialRGPT_qa/csvs"
    csv_path = os.path.join(csv_dir, chunk_name + ".csv")
    os.makedirs(csv_dir, exist_ok=True)

    oss_path = "guoxin/Spatial/datasets/spatialRGPT_qa/images/" + chunk_name + "/"
    prefix = "https://city-brain-vendor.oss-cn-hangzhou.aliyuncs.com/" + oss_path

    with open(csv_path,'w',encoding='utf8',newline='') as w_op:
        csv_writer = csv.writer(w_op)
        csv_writer.writerow(["image", "dialogue"])

        for file_name in tqdm.tqdm(sorted(os.listdir(json_dir))):
            file_path = os.path.join(json_dir, file_name)
            
            with open(file_path, "r") as r_op:
                data = json.load(r_op)
                
                data_sample = random.sample(data, 1)
                for datum in data_sample:
                    image_path = os.path.join(prefix, datum["image_path"])
                    del datum["image_path"]
                    dialogue = str(datum)

                    csv_writer.writerow([image_path, dialogue])

    ## oss
    file_list = get_image_list(oss_path, update=True)
    print(file_list)



if __name__ == "__main__":
    chunk_list = ["00000000"]
    for chunk_name in chunk_list:
        run(chunk_name)



