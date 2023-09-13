import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from inference_utils import VideoReader, VideoWriter
import torch
import torch
from model.model import MattingNetwork
import os

model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))

os.makedirs('D:\RobustVideoMatting\\submit\motion', exist_ok=True)

bgr = torch.tensor([0, 0, 0]).view(3, 1, 1).cuda()  # Green background.
rec = [None] * 4  # Initial recurrent states.
downsample_ratio = 0.25  # Adjust based on your video.
for j in ['static\\','motion\\']:
    for i in glob.glob('D:\RobustVideoMatting\\test\\'+ j + '\*.mp4'):

        reader = VideoReader(i, transform=ToTensor())
        writer = VideoWriter('D:\RobustVideoMatting\\submit\\'+ j + i.split('\\')[-1], frame_rate=25)

        with torch.no_grad():
            reader_bar = tqdm(DataLoader(reader))
            reader_bar.set_description('Processing:')
            for src in reader_bar:  # RGB tensor normalized to 0 ~ 1.
                fgr, pha, *rec = model(src.cuda(), *rec, downsample_ratio)  # Cycle the recurrent states.
                com = fgr * pha + bgr * (1 - pha)  # Composite to green background.
                writer.write(com)  # Write frame.

        print(i+'加载完成')