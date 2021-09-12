
import torch
import torch.nn as nn
import torchvision.models as models
import json

import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

def model_fn(model_dir):
    
    
    model = models.resnet50(pretrained=True)

    _ = model.eval()

    modules=list(resnet50.children())[:-1]
    model=nn.Sequential(*modules)
    for p in model.parameters():
        p.requires_grad = False

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    return model

def input_fn(request_body, request_content_type):
    assert request_content_type=='application/json'
    data = json.loads(request_body)['inputs']
    data = torch.tensor(data, dtype=torch.float32, device=device)
    return data

def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = model(input_object)
    return prediction

def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)