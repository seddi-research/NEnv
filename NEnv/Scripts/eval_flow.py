import os
import sys
from os import path

import numpy as np
import torch

from os import path

# directory reach
directory = path.path(__file__).abspath()

# Allowing imports for parent classes
sys.path.append(directory.parent)
sys.path.append(directory.parent.parent)
sys.path.append(directory.parent.parent.parent)

from NEnv.Models.NEnv import NEnv
from nsf import nn as nn_
from nsf import utils
from nsf.nde import flows, transforms
from nsf.nde.distributions.uniform import TweakedUniform


def create_base_transform(i):
    return transforms.PiecewiseRationalQuadraticCouplingTransform(
        mask=utils.create_alternating_binary_mask(
            features=2,
            even=(i % 2 == 0)
        ),
        transform_net_create_fn=lambda in_features, out_features:
        nn_.ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=200,
            num_blocks=2,
            dropout_probability=0.0,
            use_batch_norm=True,
        ),
        num_bins=200,
        apply_unconditional_transform=False,
    )


cwd = os.getcwd()

device = torch.device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# CHANGE THE PATH TO THE DESIRED .NENV file
# For example: path =r"/skylit_garage_4k/network-f.nenv"
# You can download trained models from https://javierfabre.com/nenv/

path = ""
nenv = NEnv('', load_envmap=False)

nenv.load(path)
dim = 2
distribution = TweakedUniform(
    low=torch.zeros(dim),
    high=torch.ones(dim)
)
transform = transforms.CompositeTransform([
    create_base_transform(i) for i in range(2)
])
flow = flows.Flow(transform, distribution).to(device)
flow = nenv.define_model(nenv._model_path)
flow.eval().cuda()

# For pdf evaluation at position x,y:
x = 0.01
y = 0.86

xy = torch.tensor([x, y]).float().unsqueeze(0).to(device)
log_pdf = flow.log_prob(xy).detach().cpu().numpy()
pdf = np.exp(log_pdf)
print('PDF Value:')
print(pdf)

# For sampling:
NUMBER_OF_SAMPLES = 10000
samples = flow.sample(NUMBER_OF_SAMPLES).squeeze().detach().cpu().numpy()
print('Samples:')
print(samples)
