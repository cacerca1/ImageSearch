from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
from PIL import Image
import base64
from io import BytesIO
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import pickle 

app = Flask(__name__)
 
#UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        k = int(request.form['text'])

        filename = secure_filename(file.filename) 
        flash(file.filename)
        im = Image.open(file.stream)
        data = BytesIO()
        im.save(data, "JPEG")
        enc_img = base64.b64encode(data.getvalue())
        dec_img = enc_img.decode('utf-8')
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', img_data=dec_img, images=infer(enc_img, k))#[dec_img for _ in range(k)])
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

def get_model():
    resnet50 = models.resnet50(pretrained=False)

    resnet50.load_state_dict(torch.load('/home/ccaceresgarcia/Documents/Projects/image_search/resnet50-19c8e357.pth')) # path of your weights

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

    with open('/home/ccaceresgarcia/Documents/Projects/image_search/ImageSearch/filenames.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        loaded_filenames = pickle.load(f)

    # load the model from disk
    loaded_model = pickle.load(open('/home/ccaceresgarcia/Documents/Projects/image_search/ImageSearch/knnpickle_file.pickle', 'rb'))

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
    
if __name__ == "__main__":
    app.run()
