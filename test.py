import torch
import logging
root_logger = logging.getLogger()
from models.internimage import InternImage
from utils.get_model import get_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = get_model('imagenet', 'internimageemb', 1, 4, 'embtiny', 'classification', logger=root_logger)[0][0].to(device='cuda')
x = torch.rand((32,3,512,512), device='cuda')
out = model(x)