import sys
sys.path.append(".")  # 这里将scripts目录的父目录添加到路径中

import os

# 获取当前脚本的绝对路径
script_path = os.path.abspath(__file__)
# 获得当前脚本所在目录的父目录（即train目录所在的位置）
parent_dir = os.path.dirname(os.path.dirname(script_path))
# 将父目录添加到Python的搜索路径中
sys.path.append(parent_dir)
import argparse
from datasets import Dataset, load_dataset, load_from_disk

import torch
from diffusers import AutoencoderKL

from train.utils import preprocess_train_image_fn
from PIL import Image

def get_image_name(url):
    # 分割URL以获取文件名（不包括后缀）
    file_name = url.split('/')[-1].split('.')[0]

    # 假设ID是文件名中的数字部分
    id = ''.join(filter(str.isdigit, file_name))

    # 提取后缀名
    subfix = url.split('.')[-1]
    if len(subfix) > 4:
        subfix = "uk"

    return f"{id}.{subfix}"        

def get_new_dataset_generator(dataset, vae_transform_fn, vae, args):
    def sample_generator():
        with torch.no_grad():
            for sample in dataset:
                caption = sample['TEXT']
                image = sample['URL']
                url = image
                image = get_image_name(image)
                try:
                    image = Image.open(image)
                    image, original_size, crop_top_left, target_size = vae_transform_fn(image)
                    image = image.unsqueeze(0)
                    image = image.to(args.dtype)
                    image = image.cuda()
                    image_feature = vae.encode(image).latent_dist.parameters[0].to("cpu")

                    yield {
                        "caption": caption,
                        "image_feature": image_feature,
                        "original_size": original_size,
                        "crop_top_left": crop_top_left,
                        "target_size": target_size,
                    }
                except Exception:
                    with open('error.txt', 'a') as f:
                        f.write(url + '\n')
                    #print('error')
                    continue
    return sample_generator

def convert(args):
    # if os.path.exists(args.original_dataset_name_or_path):
    #     dataset = load_from_disk(args.original_dataset_name_or_path)
    # else:
    #     dataset = load_dataset(args.original_dataset_name_or_path)
    dataset = load_dataset('csv', data_files=args.original_dataset_name_or_path)

    if args.split is not None:
        dataset = dataset[args.split]
    dataset = dataset['train']

    vae_transform_fn = preprocess_train_image_fn(args)
    vae = AutoencoderKL.from_pretrained(
            args.vae_model_path,
            subfolder="unet",
            cache_dir=args.cache_dir,
    ).to(args.dtype).cuda()

    ds = Dataset.from_generator(
            get_new_dataset_generator(dataset, vae_transform_fn, vae, args),
            cache_dir=args.cache_dir,
            num_proc=args.num_proc,
    )

    ds.save_to_disk(args.output_dataset_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dataset_path", required=True)
    parser.add_argument("--vae_model_path", required=True)
    parser.add_argument("--original_dataset_name_or_path", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--caption_column", default="caption")
    parser.add_argument("--image_column", default="image")
    parser.add_argument("--image_resolution", default=512, type=int)
    parser.add_argument("--image_center_crop", default=True)
    parser.add_argument("--image_random_flip", default=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--num_proc", default=None, type=int)
    args = parser.parse_args()

    if args.dtype.lower() in ["fp16", "float16"]:
        args.dtype = torch.float16
    elif args.dtype.lower() in ["fp32", "float32"]:
        args.dtype = torch.float32
    else:
        raise RuntimeError(f"Unsupported dtype {args.dtype}")

    convert(args)
