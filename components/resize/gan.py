import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

import numpy as np
import math
from PIL import Image
from tqdm import tqdm
import shutil

class Gan():
    def __init__(self,gpu_id=None):
        model_name = 'RealESRGAN_x4plus'
        model_path = None
        denoise_strength = 0.5
        tile = 0
        tile_pad = 10
        pre_pad = 0
        fp32 = False
        if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
        elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
        elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
        elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]

        # determine model paths
        if model_path is not None:
            model_path = model_path
        else:
            model_path = os.path.join('weights', model_name + '.pth')
            if not os.path.isfile(model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in file_url:
                    # model_path will be updated
                    model_path = load_file_from_url(
                        url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

        # use dni to control the denoise strength
        dni_weight = None
        if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            dni_weight = [denoise_strength, 1 - denoise_strength]

        # restorer
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=not fp32,
            gpu_id=gpu_id)
    
    def optimize(self,image,outscale=4,face_enhance=False):
        image = np.array(image,dtype=np.uint8)
        if face_enhance:  # Use GFPGAN for face enhancement
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self.upsampler)
        try:
            if face_enhance:
                _, _, output = face_enhancer.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = self.upsampler.enhance(image, outscale=outscale)
            return np.array(output,dtype=np.uint8)
        except RuntimeError as error:
            print('Error', error)

    def resize(self,image,size=1024):
        height,width,_ = np.array(image,dtype=np.uint8).shape
        upscale = 1024/min(width,height)
        upscale = math.ceil(upscale*100)/100
        # 放大
        if upscale>1:
            image2 = self.optimize(image,upscale)
        else:
            if width<height:
                image.resize((1024,math.ceil(1024/width*height)))
            else:
                image.resize((math.ceil(1024/height*width),1024))
            image2 = np.array(image,dtype=np.uint8)
        height2,width2,_ = image2.shape
        w0 = int((width2-size)/2)
        h0 = int((height2-size)/2)
        # 裁剪
        image3 = image2[h0:h0+size,w0:w0+size,:]
        return image3
    
    def resize256(self,image):
        return image.resize((256,256))
    
    def resizeDir(self,src_dir,rst_dir):
        if os.path.exists(rst_dir):
            shutil.rmtree(rst_dir)
        os.makedirs(rst_dir)
        for filename in tqdm(os.listdir(src_dir)):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(src_dir, filename)
                try:
                    pil_image = Image.open(img_path)
                    image_1024 = self.resize(pil_image)
                    image_1024 = Image.fromarray(image_1024)
                    save_path = os.path.join(rst_dir, filename)
                    image_1024.save(save_path)
                    pil_image.close()
                except Exception as e:
                    print(e)
    
if __name__ == '__main__':
    gan = Gan()
    