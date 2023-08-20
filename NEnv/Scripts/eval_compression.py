import os
import sys
from os import path

import torch

# directory reach
directory = path.path(__file__).abspath()

# Allowing imports for parent classes
sys.path.append(directory.parent)
sys.path.append(directory.parent.parent)
sys.path.append(directory.parent.parent.parent)

from NEnv.Models.NEnv_Compression import NEnv_Compression

cwd = os.getcwd()

device = torch.device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# CHANGE THE PATH TO THE DESIRED .NENV file
# For example: path =r"/skylit_garage_4k/network-c.pth"
# You can download trained models from https://javierfabre.com/nenv/

path = nenv = NEnv_Compression('', load_envmap=False)

nenv.load(path)

# Load model
model = nenv.define_model(nenv._model_path).half().eval()

# For RGB evaluation at position x,y:
x = 0.01
y = 0.86

xy = torch.tensor([x, y]).float().unsqueeze(0).to(device)
rgb = model(xy)
