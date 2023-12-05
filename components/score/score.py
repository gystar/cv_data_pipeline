from PIL import Image
import os
import shutil
from warnings import filterwarnings
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path



from concurrent.futures import ThreadPoolExecutor, as_completed


import clip
from PIL import Image

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

class Score():
    def __init__(self):
        torch.manual_seed(2618)
        # set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # set mlp
        model = MLP(768)
        curren_dir = Path(__file__).parent
        s = torch.load(curren_dir / "./sac+logos+ava1-l14-linearMSE.pth", map_location=self.device)
        model.load_state_dict(s)
        self.model = model.to(self.device)
        self.model.eval()
        # set clip
        model2, preprocess = clip.load("ViT-L/14", device=self.device)  #RN50x64   
        self.clip = model2
        self.preprocess = preprocess
        
        
    def score_image(self, img_path:Path, aspect_limit, threshold, rst_dir:Path):                           
        try:
            pil_image = Image.open(img_path)                
            # 宽高比差太多，删除
            width,height = pil_image.size
            if height/width>aspect_limit or width/height>aspect_limit:
                #os.remove(img_path)
                return 0
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.clip.encode_image(image)
                im_emb_arr = normalized(image_features.cpu().detach().numpy())
                
                prediction = self.model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor))
                score = float(prediction[0][0])
                score = round(score,8)
            
            file_name = str(img_path.name).split(".")[0]
            save_path = rst_dir / f'{file_name}_{score}.png'
            pil_image.save(save_path)
            pil_image.close()
            return 1
        except Exception as e:
            print(f"无法处理文件 {img_path}: {e}")
            return 0
            
    def predict(self,src_dir,rst_dir,aspect_limit=2,threshold=4.5):
        if os.path.exists(rst_dir):
            shutil.rmtree(rst_dir)
        os.makedirs(rst_dir)        
        executor = ThreadPoolExecutor(max_workers=16)   
        files = os.listdir(src_dir)
        futures=[]
        total_count = len(files)
        processed_count = 0
        for i, filename in tqdm( enumerate(files)):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = Path(src_dir) /filename
                futures.append(executor.submit(self.score_image, img_path, aspect_limit, threshold, Path(rst_dir)))          
                # 每1000条等待执行完成，检查一下进度，不要等全部sumbit，那样会导致内存溢出
                if len(futures) == 1000 or i+1 == len(files):
                    selected_count = 0                    
                    for f in as_completed(futures): 
                        selected_count += f.result()
                    processed_count += len(futures)
                    percent = selected_count/len(futures) * 100
                    total_percent = processed_count / float(total_count) *100
                    print(f"[{total_percent:.2f}% / {total_count}]==========processed {percent}% images of {len(futures)} images.=============")                                   
                    futures = []


if __name__ == '__main__':
    src_dir = '/data/lz/improved-aesthetic-predictor/test_src'
    rst_dir = '/data/lz/improved-aesthetic-predictor/test_rst'
    filter = Score()
    filter.predict(src_dir,rst_dir)
    
    














