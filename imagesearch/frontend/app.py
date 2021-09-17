from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from requests import models
from werkzeug.utils import secure_filename
from PIL import Image
import base64
from io import BytesIO

from ..models.endpoint import LambdaSearch

app = Flask(__name__)
 
 
app.secret_key = "secret key"
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
        return render_template('index.html', img_data=dec_img, images=LambdaSearch().infer(dec_img, k))
        # return render_template('index.html', img_data=dec_img, images=infer(enc_img, k))
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

if __name__ == "__main__":
    app.run()
