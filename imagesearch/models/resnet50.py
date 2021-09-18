
import base64
from io import BytesIO
import numpy as np
import pickle 
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms

from .model import ExtractorSearch
from ..utils import open_image
from dotenv import load_dotenv

load_dotenv()
KNNMODEL = os.getenv('KNNMODEL')
FILENAMES = os.getenv('FILENAMES')

class Resnet50Search(ExtractorSearch):
    def get_model(self):
        failed = True
        while failed:
            try:
                resnet50 = models.resnet50(pretrained=True)
                failed = False
            except:
                pass

        #resnet50.load_state_dict(torch.load(MODELWEIGHTS)) # path of your weights

        _ = resnet50.eval()
        _ = resnet50.cuda()

        modules=list(resnet50.children())[:-1]
        resnet50=nn.Sequential(*modules)
        for p in resnet50.parameters():
            p.requires_grad = False

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

        self.model = resnet50.to(self.device)
        
    def get_embedding(self, enc_img):
        
        im_bytes = base64.b64decode(enc_img)   # im_bytes is a binary image
        im_file = BytesIO(im_bytes)  # convert image to file-like object
        image = open_image(im_file)

        self.get_model()
        
        im = np.asarray(image)# convert image to numpy array
        img = self.transform(im) # convert to tensor
        img = torch.unsqueeze(img, 0)
        img = img.to(self.device)
        
        with torch.no_grad():
            feature = self.model(img)
        
        return feature
        
    def infer(self, enc_img, k):
        
        feature = self.get_embedding(enc_img)
            
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
            im = open_image(loaded_filenames[match])

            data = BytesIO()
            im.save(data, "JPEG")
            enc_img = base64.b64encode(data.getvalue())
            dec_img = enc_img.decode('utf-8')
        
            similar_images.append(dec_img)
            
        return similar_images
        