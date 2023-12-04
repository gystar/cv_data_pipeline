import os
from tqdm import tqdm
from PIL import Image
import json
import torch
import numpy as np
from .models.modeling_emu import Emu
import threading

image_placeholder = "[IMG]" + "<image>" * 32 + "[/IMG]"
image_system_msg = "You will be presented with an image: [IMG]ImageContent[/IMG]. You will be able to see the image after I provide it to you. Please answer my questions based on the given image."

class Summary():
    def __init__(self,map_file=None,devices=['cuda:0']):
        self.summary_map = []
        self.devices = devices
        self.emu_models = []
        self.lock = threading.Lock()
        if map_file:
            try:
                with open(map_file, 'r+') as f:
                    self.summary_map = json.load(f)
                f.close()
            except Exception as e:
                pass
        for device in devices:
            emu_model = self.prepare_model()
            self.emu_models.append(emu_model.to(device).to(torch.bfloat16))
        
    def process_img(self,img_path=None, img=None, device='cuda:0'):
        assert img_path is not None or img is not None, "you should pass either path to an image or a PIL image object"
        width, height = 224, 224
        OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
        OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
        
        if img_path:
            img = Image.open(img_path)
        img = img.convert("RGB")
        img = img.resize((width, height))
        img = np.array(img) / 255.
        img = (img - OPENAI_DATASET_MEAN) / OPENAI_DATASET_STD
        img = torch.tensor(img).to(device).to(torch.float)
        img = torch.einsum('hwc->chw', img)
        img = img.unsqueeze(0)
        return img
        
    def prepare_model(self):
        with open(f'/data/lz/Emu/models/Emu-14B.json', "r", encoding="utf8") as f:
            model_cfg = json.load(f)
        print(f"=====> model_cfg: {model_cfg}")
        model = Emu(**model_cfg, cast_dtype=torch.float)

        if True:
            print('Patching LoRA...')
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model.decoder.lm = get_peft_model(model.decoder.lm, lora_config)

        # print(f"=====> loading from ckpt_path {args.ckpt_path}")
        ckpt = torch.load("/data/lz/models/Emu-instruct.pt", map_location="cpu")
        if 'module' in ckpt:
            ckpt = ckpt['module']
        msg = model.load_state_dict(ckpt, strict=False)
        model.eval()
        # print(f"=====> get model.load_state_dict msg: {msg}")

        return model

    def inference(self,img,input,bias=0):
        image = self.process_img(img=img, device=self.devices[bias])
        instruct = True
        
        image_list = [image]
        text_sequence = image_placeholder + input
        if instruct:
            prompt = f"{image_system_msg} [USER]: {text_sequence} [ASSISTANT]:".strip()
        else:
            prompt = text_sequence

        # print(f"===> prompt: {prompt}")

        samples = {"image": torch.cat(image_list, dim=0), "prompt": prompt}

        output_text = self.emu_models[bias].generate(
            samples,
            max_new_tokens=128,
            num_beams=5,
            length_penalty=0.0,
            repetition_penalty=1.0,
            device=self.devices[bias]
        )[0].strip()

        return output_text

    def run(self,src_dir,cos_dir,map_file,bias=0):
        # 调多模态模型对图片进行描述
        n = len(self.devices)
        for filename in tqdm(os.listdir(src_dir)[bias::n]):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                img_path = os.path.join(src_dir, filename)
                img_url = cos_dir+filename
                if any(e['local_path'] == img_path for e in self.summary_map):
                    continue
                try:
                    pil_image = Image.open(img_path)
                    # call model to summary
                    summary = self.inference(pil_image,'Summarize the picture in detail about 70 words',bias)
                    #
                    self.lock.acquire()
                    self.summary_map.append({
                        'local_path': img_path,
                        'cos_url': img_url,
                        'summary': summary
                    })
                    self.lock.release()
                except Exception as e:
                    print('Summary Error: {}'.format(e))
                with open(map_file, 'w+') as f:
                    json.dump(self.summary_map, f, indent=2)
                f.close()
  
    def run_n(self,src_dir,cos_dir,map_file):
        pool = []
        for i,_ in enumerate(self.devices):
            t = threading.Thread(target=self.run,args=(src_dir,cos_dir,map_file,i))
            pool.append(t)
            t.start()
            
        for t in pool:
            t.join()
            