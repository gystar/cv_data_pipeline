import os
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from tqdm import tqdm

# 腾讯云 COS 配置
secret_id = 'AKIDfQzjLlwvfDcniUBopDgAe6qkhfXp297s'
secret_key = 'd6OceT7ZMzPpAKRZkyocRoCf6UtBbCJQ'
region = 'ap-guangzhou'
bucket = 'dragondiffusion-1316760375'


# 创建 COS 配置
config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
client = CosS3Client(config)

# 要上传的本地目录
local_directory = '/data/lz/data_pipeline/data/test_result'
# 目标文件夹路径
folder_path = 'test_result/'

def upload(local_directory,folder_path):
    # 检查目标文件夹是否存在
    response = client.list_objects(Bucket=bucket, Prefix=folder_path)

    # 如果目标文件夹不存在，则创建它
    if 'Contents' not in response:
        response = client.put_object(
            Bucket=bucket,
            Key=folder_path,
            Body=''
        )

        # 检查创建是否成功
        # if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        #     print(f'Folder created: {folder_path}')
        # else:
        #     print(f'Failed to create folder: {folder_path}')
        #     exit(1)
    else:
        print(f'Folder already exists: {folder_path}')

    # 遍历文件列表并上传
    for file_name in os.listdir(local_directory):
        file_path = os.path.join(local_directory, file_name)
        if os.path.isfile(file_path):
            # 获取文件大小
            file_size = os.path.getsize(file_path)

            # 初始化分块上传
            response = client.create_multipart_upload(Bucket=bucket, Key=folder_path + file_name)
            upload_id = response['UploadId']

            # 上传分块
            part_number = 1
            parts = []
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=file_name) as pbar:
                with open(file_path, 'rb') as f:
                    while True:
                        data = f.read(50 * 1024 * 1024)  # 每次读取 50MB 数据
                        if not data:
                            break
                        response = client.upload_part(
                            Bucket=bucket,
                            Key=folder_path + file_name,
                            PartNumber=part_number,
                            UploadId=upload_id,
                            Body=data
                        )
                        part = {
                            'PartNumber': part_number,
                            'ETag': response['ETag']
                        }
                        parts.append(part)
                        part_number += 1
                        pbar.update(len(data))

            # 完成分块上传
            response = client.complete_multipart_upload(
                Bucket=bucket,
                Key=folder_path + file_name,
                UploadId=upload_id,
                MultipartUpload={'Part': parts}
            )

            # 检查上传是否成功
            if 'Location' in response and response['Location'].startswith('http'):
                print(f'Uploaded: {folder_path + file_name}')
            else:
                print(f'Upload failed: {folder_path + file_name}')