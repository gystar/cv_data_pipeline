from tqdm import tqdm
import json
import os
import openai
import threading
import time


# OPENAI_API_BASE = "https://one-api.bltcy.top/v1"
# openai.api_key = "sk-x7BiHoSfjxsxzmoi066f6cB0Ac014b1494843cEe3102AeF8"
# openai.api_base = "https://api.chatgpt-3.vip/v1"
# openai.api_key = "sk-w68eMiLURq9Q1GukA775B7E2F54e4699A6F97e3f88BfEc58"
openai.api_base = "https://api.myhispreadnlp.com/v1"
# openai.api_key = "O76y209bP58Urq9RWOE21SO58hlObVwL4jFenR71xmUFv5qt"
openai.api_key = "M0r73BW3b8QKtdQmh6tAI1K86Y66LUvVD2uDOGe5q2rU92M1"


instruct = """You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say in square brackets. For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" will trigger your partner bot to output an image of a forest morning, as described.
You will be prompted by people looking to create detailed, amazing images. The way to accomplish this
is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:
- Your return includes 3 parts: detailed Chinese description,detailed English description and SD prompt words that can be used for Stable Diffuson to draw high quality images.
- The 3 parts are seperated by '<>'
- For example: '美丽的水墨山水画，山顶上有一只仙鹤和一位身着青云道袍的老者 <> Beautiful ink landscape painting, on the top of the mountain there is a crane and an old man dressed in green clouds and Taoist robes <> Beautiful, ink, landscape, painting, mountain, summit, celestial crane, elder, azure, Daoist robes.'
- The Chinese and English descriptions should be as detailed as possible.
- The Stable Diffusion prompt words should be about 75 in length.
- The Stable Diffusion prompt words should be high quality and accuracy that can help produce high quality images.
"""

class Summaryx():
    def __init__(self,map_file=None):
        self.lock = threading.Lock()
        self.summary_map = []
        if map_file:
            try:
                with open(map_file, 'r+') as f:
                    self.summary_map = json.load(f)
                f.close()
            except Exception as e:
                pass
    
    def summary(self,img_url):
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": instruct}]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url":  img_url
                        },
                        {
                            "type": "text",
                            "text": "Please return three descriptions as specified,seperate them with '<>'"
                        }
                    ]
                }
            ],
            max_tokens=1500
        )
        
        return response.choices[0]['message']['content']

    def summaryDir(self,src_dir,cos_dir,map_file,n=1,bias=0):
        # 调多模态模型对图片进行描述
        filenames = sorted(os.listdir(src_dir))[bias::n]
        filenames = list(filter(lambda x: not any([e for e in self.summary_map if x in e['local_path']]) ,filenames))
        for filename in tqdm(filenames):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(src_dir, filename)
                img_url = cos_dir+filename
                # if any(e['local_path'] == img_path for e in self.summary_map):
                #     continue
                count = 0
                while True:
                    if count> 20:
                        break
                    try:
                        # call model to summary
                        result = self.summary(img_url)
                        summary = ''
                        for chunk in result:
                            summary += chunk
                        #
                        self.lock.acquire()
                        self.summary_map.append({
                            'local_path': img_path,
                            'cos_url': img_url,
                            'desc_en': summary.split('<>')[0].strip(),
                            'desc_cn': summary.split('<>')[1].strip(),
                            'tags': summary.split('<>')[2].strip()
                        })
                        self.lock.release()
                    except Exception as e:
                        # print('Summary Error: {}'.format(e))
                        # print('Try again 5s later')
                        # print(img_url)
                        count += 1
                        time.sleep(3)
                    else:
                        with open(map_file, 'w+') as f:
                            json.dump(self.summary_map, f, indent=2)
                        f.close()
                        break
                    
    def summaryDir_nWorkers(self,src_dir,cos_dir,map_file,n):
        pool = []
        for i in range(n):
            t = threading.Thread(target=self.summaryDir,args=(src_dir,cos_dir,map_file,n,i))
            pool.append(t)
            t.start()
            time.sleep(5)
            
        for t in pool:
            t.join()