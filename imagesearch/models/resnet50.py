
from PIL import Image
import base64
from io import BytesIO
import numpy as np
import pickle 

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms

KNNMODEL = '/home/ccaceresgarcia/Documents/Projects/image_search/ImageSearch/knnpickle_file.pickle'
MODELWEIGHTS = '/home/ccaceresgarcia/Documents/Projects/image_search/resnet50-19c8e357.pth'
FILENAMES = '/home/ccaceresgarcia/Documents/Projects/image_search/ImageSearch/filenames.pickle'

def get_model():
    resnet50 = models.resnet50(pretrained=False)

    resnet50.load_state_dict(torch.load(MODELWEIGHTS)) # path of your weights

    _ = resnet50.eval()
    _ = resnet50.cuda()

    modules=list(resnet50.children())[:-1]
    resnet50=nn.Sequential(*modules)
    for p in resnet50.parameters():
        p.requires_grad = False

    transform = transforms.Compose([transforms.ToTensor()])

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    resnet50 = resnet50.to(device)
    
    return resnet50, transform, device

def infer(enc_img, k):
    
    im_bytes = base64.b64decode(enc_img)   # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  # convert image to file-like object
    image = Image.open(im_file)   # img is now PIL Image object

    resnet50, transform, device = get_model()
    
    im = np.asarray(image)# convert image to numpy array
    img = transform(im) # convert to tensor
    img = torch.unsqueeze(img, 0)
    img = img.to(device)
    
    with torch.no_grad():
        feature = resnet50(img)

    with open(FILENAMES, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        loaded_filenames = pickle.load(f)

    # load the model from disk
    loaded_model = pickle.load(open(KNNMODEL, 'rb'))

    flat_feature = feature.cpu().detach().numpy().reshape(-1)
    distances, indices = loaded_model.kneighbors([flat_feature])
    
    similar_images = []
    for i in range(k):
        # load the image
        match = indices[0][i]
        print(loaded_filenames[match])
        im = Image.open(loaded_filenames[match])

        data = BytesIO()
        im.save(data, "JPEG")
        enc_img = base64.b64encode(data.getvalue())
        dec_img = enc_img.decode('utf-8')
    
        similar_images.append(dec_img)
        
    return similar_images
    