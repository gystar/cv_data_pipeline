import os
# os.environ['https_proxy'] = '127.0.0.1:7890'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from utils.batch_download import listCurrentDir,downLoadDirFromCos
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import json

from components import *
from utils.upload import upload


prefix = 'https://dragondiffusion-1316760375.cos.ap-guangzhou.myqcloud.com/'

def get_cos_img_list(cos_dir):
    all = listCurrentDir(cos_dir)
    cos_img_list = list(filter(lambda x: any([a in x['Key'] for a in ['.jpg','.png','.jpeg']]),all))
    cos_img_list = sorted(cos_img_list,key=lambda file_info: file_info["Key"])
    return cos_img_list

def download(cos_dir,save_dir):
    save_dir = save_dir+'images/' if save_dir[-1]=='/' else save_dir+'/images/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cos_img_list0 = get_cos_img_list(cos_dir)
    downloaded = os.listdir(save_dir)
    cos_img_list = []
    print('Check downloaded:')
    for item in tqdm(cos_img_list0):
        if not any([f for f in downloaded if (f.replace('.jpg','').replace('.jpeg','').replace('.png','') in item['Key'])]):
            cos_img_list.append(item)
    print('Downloading:')
    for img_info in tqdm(cos_img_list):
        try:
            cos_url = prefix + img_info['Key']
            local_path = save_dir+img_info['Key'].split('/')[-1].replace('.jpg','.png').replace('.jpeg','.png')
            response = requests.get(cos_url)
            img = Image.open(BytesIO(response.content)).convert('RGBA')
            img.save(local_path)
        except:
            pass
        # map_json.append({
        #     'cos_url': cos_url,
        #     'local_path': local_path
        # })
        # if len(map_json) % 10 == 0:
        #     with open(download_map_file,'w+') as f:
        #         json.dump(map_json,f)
        #     f.close()
    # with open(download_map_file,'w+') as f:
    #     json.dump(map_json,f)
    # f.close()
    
class Data_Pipeline():
    def __init__(self):
        pass
    
    def download_cos(self,cos_dir,local_dir):
        download(cos_dir,local_dir)
    
    def upload_cos(self,local_dir,cos_dir):
        upload(local_dir,cos_dir)
        
    def run(self,cos_src,local_dir='/data/lz/data/',cos_save_dir="data_pipeline/",n=1):
        # if cos_src[-1]=='/':
        #     cos_src = cos_src[:-1]
        # name = cos_src.split('/')[-1]
        # local_dir += name
        # cos_save_dir += (name+'/')
        # print('Start Download')
        # self.download_cos(cos_src,local_dir)
        # print('Start Score')
        # self.score = Score()
        # self.score.predict(local_dir+'/images',local_dir+'/scored')
        # del self.score
        # print('Start Resize')
        # self.gan = Gan()
        # self.gan.resizeDir(local_dir+'/scored',local_dir+'/resized')
        # del self.gan
        # print('Upload COS')
        # self.upload_cos(local_dir+'/resized',cos_save_dir)
        # print('Start Summary by gpt4v')
        # self.summary = Summaryx(local_dir+'/result_map_gpt4v.json')
        # self.summary.summaryDir_nWorkers(local_dir+'/resized',prefix+cos_save_dir,local_dir+'/result_map_gpt4v.json',n)
        # del self.summary
        # print('Start Summary by emu')
        # self.summary = Summary(dir+'/result_map.json',devices=['cuda:7'])
        # self.summary.run_n(dir+'/resized',prefix+cos_dir,dir+'/result_map_emu.json')
        # del self.summary
        print('Start Classify')
        self.classify = Classify()
        # self.classify.predict(dir+'/result_map.json',dir+'/result_map_with_class.json')
        self.classify.predict('/data/lz/data/data_pipeline_test/result_map_gpt4v.json','/data/lz/data/data_pipeline_test/result_map_gpt4v_with_class.json')
        del self.classify

if __name__ == '__main__':
    # download('/data/lz/data/data_pipeline_test')
    dp = Data_Pipeline()
    dp.run('数据集/视觉中国/2、插画/插画/6、插画描线风/',n=10)