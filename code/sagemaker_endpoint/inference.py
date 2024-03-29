
import torch
import torch.nn as nn
import torchvision.models as models
import json
import base64
from io import BytesIO

import os
import numpy as np
from PIL import Image

def model_fn(model_dir):
    """Herlper function for sagemaker endpoint to get the model.
    """
    model = models.resnet50(pretrained=True)

    _ = model.eval()

    modules=list(model.children())[:-1]
    model=nn.Sequential(*modules)
    for p in model.parameters():
        p.requires_grad = False

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    return model

def input_fn(request_body, request_content_type='application/json'):
    """Helper function for sagemaker endpoint to process in input before passing it to the model for inference.
    """
    if request_content_type =='application/json':
        data = json.loads(request_body)
        data = data['inputs']
        
        im_bytes = base64.b64decode(data)   # im_bytes is a binary image
        im_file = BytesIO(im_bytes)  # convert image to file-like object
        image = Image.open(im_file)   # img is now PIL Image object
        im = np.asarray(image)# convert image to numpy array
        
        # replicate transforms.to_tensor
        im = np.moveaxis(im, -1, 0) # transpose to channels first
        data = torch.tensor(im, dtype=torch.float32)#, device=device)
        data /= 255
        
        return data
    raise Exception("Unsupported ContentType: %s", request_content_type)

def predict_fn(input_object, model):
    """Helper function to predict on an image using the model
    """
    if torch.cuda.is_available():
        input_object = input_object.cuda()
    input_object = torch.unsqueeze(input_object, 0)

    with torch.no_grad():
        prediction = model(input_object)
    return prediction

def output_fn(predictions, content_type):
    """Helper function to process the predictions of the model before returning to the user.
    """
    assert content_type == 'application/json'
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)