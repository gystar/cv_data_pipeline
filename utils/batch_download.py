# -*- coding=utf-8
import os
import logging
import sys
import json
import os

from qcloud_cos import CosConfig, CosServiceError
from qcloud_cos import CosS3Client
from qcloud_cos.cos_threadpool import SimpleThreadPool

# 正常情况日志级别使用 INFO，需要定位时可以修改为 DEBUG，此时 SDK 会打印和服务端的通信信息
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# 1. 设置用户属性, 包括 secret_id, secret_key, region等。Appid 已在 CosConfig 中移除，请在参数 Bucket 中带上 Appid。Bucket 由 BucketName-Appid 组成
secret_id = 'AKIDMYWIIDappdDZhdLo35vXv76uTouteCXg'   # 用户的 SecretId，建议使用子账号密钥，授权遵循最小权限指引，降低使用风险。子账号密钥获取可参见 https://cloud.tencent.com/document/product/598/37140
secret_key = '0mKew0qSpytUxCD1d2EE1cBXhiW7WiiV'  # 用户的 SecretKey，建议使用子账号密钥，授权遵循最小权限指引，降低使用风险。子账号密钥获取可参见 https://cloud.tencent.com/document/product/598/37140
region = 'ap-guangzhou'      # 替换为用户的 region，已创建桶归属的 region 可以在控制台查看，https://console.cloud.tencent.com/cos5/bucket
                           # COS 支持的所有 region 列表参见 https://cloud.tencent.com/document/product/436/6224
token = None               # 如果使用永久密钥不需要填入 token，如果使用临时密钥需要填入，临时密钥生成和使用指引参见 https://cloud.tencent.com/document/product/436/14048
scheme = 'https'           # 指定使用 http/https 协议来访问 COS，默认为 https，可不填

config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token, Scheme=scheme)  # 获取配置对象
client = CosS3Client(config)

# 用户的 bucket 信息
test_bucket = 'dragondiffusion-1316760375'
start_prefix = '/数据集/视觉中国/1、人物/人物/4、卡通3D角色'
# 对象存储依赖 分隔符 '/' 来模拟目录语义，
# 使用默认的空分隔符可以列出目录下面的所有子节点，实现类似本地目录递归的效果,
# 如果 delimiter 设置为 "/"，则需要在程序里递归处理子目录
delimiter = ''


# 列出当前目录子节点，返回所有子节点信息
def listCurrentDir(prefix):
    file_infos = []
    sub_dirs = []
    marker = ""
    count = 1
    while True:
        response = client.list_objects(test_bucket, prefix, delimiter, marker)
        # 调试输出
        # json_object = json.dumps(response, indent=4)
        # print(count, " =======================================")
        # print(json_object)
        count += 1

        if "CommonPrefixes" in response:
            common_prefixes = response.get("CommonPrefixes")
            sub_dirs.extend(common_prefixes)

        if "Contents" in response:
            contents = response.get("Contents")
            file_infos.extend(contents)

        if "NextMarker" in response.keys():
            marker = response["NextMarker"]
        else:
            break

    print("=======================================================")

    # 如果 delimiter 设置为 "/"，则需要进行递归处理子目录，
    # sorted(sub_dirs, key=lambda sub_dir: sub_dir["Prefix"])
    # for sub_dir in sub_dirs:
    #     print(sub_dir)
    #     sub_dir_files = listCurrentDir(sub_dir["Prefix"])
    #     file_infos.extend(sub_dir_files)

    print("=======================================================")

    sorted(file_infos, key=lambda file_info: file_info["Key"])
    for file in file_infos:
        print(file)
    return file_infos


# 下载文件到本地目录，如果本地目录已经有同名文件则会被覆盖；
# 如果目录结构不存在，则会创建和对象存储一样的目录结构
def downLoadFiles(file_infos,localDir):

    pool = SimpleThreadPool()
    for file in file_infos:
        # 文件下载 获取文件到本地
        file_cos_key = file["Key"]
        localName = localDir + file_cos_key

        # 如果本地目录结构不存在，递归创建
        if not os.path.exists(os.path.dirname(localName)):
            os.makedirs(os.path.dirname(localName))

        # skip dir, no need to download it
        if str(localName).endswith("/"):
            continue

        # 实际下载文件
        # 使用线程池方式
        pool.add_task(client.download_file, test_bucket, file_cos_key, localName)

        # 简单下载方式
        # response = client.get_object(
        #     Bucket=test_bucket,
        #     Key=file_cos_key,
        # )
        # response['Body'].get_stream_to_file(localName)

    pool.wait_completion()
    return None


# 功能封装，下载对象存储上面的一个目录到本地磁盘
def downLoadDirFromCos(prefix,localDir):
    global file_infos

    try:
        file_infos = listCurrentDir(prefix)

    except CosServiceError as e:
        print(e.get_origin_msg())
        print(e.get_digest_msg())
        print(e.get_status_code())
        print(e.get_error_code())
        print(e.get_error_msg())
        print(e.get_resource_location())
        print(e.get_trace_id())
        print(e.get_request_id())

    downLoadFiles(file_infos,localDir)
    return None


if __name__ == "__main__":
    downLoadDirFromCos(start_prefix)