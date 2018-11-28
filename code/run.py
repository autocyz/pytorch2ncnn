import torch.nn as nn
import torch

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

from ReplaceDenormals import ReplaceDenormals
from ConvertModel import ConvertModel_ncnn
import torch
import pickle
import os
from functools import partial

import sys
sys.path.append("../ModelFiles/")
from classfication_cr.small_nn import *

# this project must run at pytorch0.2

# set your pytorch model_path here
model_path = '../ModelFiles/classfication_cr/augmentation11_23.pth.tar'

# pickle.load = partial(pickle.load, encoding="latin1")
# pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
print("loading model....")
ckpt = torch.load(model_path, map_location=lambda storage, loc: storage, pickle_module=pickle)
print("load model over")

model = smallA_bn()
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

print("load parameters over")

# ReplaceDenormals(model)

"""  Connnnnnnnvert!  """
ModelDir = '../ModelFiles/'
text_net, binary_weights = ConvertModel_ncnn(model, [1,3,224,224], softmax=False)

"""  Save files  """
NetName = str(model.__class__.__name__)
if not os.path.exists(ModelDir + NetName):
    os.makedirs(ModelDir + NetName)
print('Saving to ' + ModelDir + NetName)

import numpy as np

with open(ModelDir + NetName + '/' + NetName + '.param.bin', 'wb') as f:
    f.write(text_net)
with open(ModelDir + NetName + '/' + NetName + '.bin', 'wb') as f:
    for weights in binary_weights:
        for blob in weights:
            blob_32f = blob.flatten().astype(np.float32)
            blob_32f.tofile(f)

print('Converting Done.')
