from flask import Flask, flash, request, redirect, url_for, render_template
from requests import models
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import os
from ..utils import open_image

app = Flask(__name__)
 
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
 
app.config['method'] = os.environ.get('RUNMETHOD','local') #sys.argv[1]
if app.config['method'] == 'local':
    from imagesearch.models.resnet50 import Resnet50Search
else:
    from imagesearch.models.endpoint import LambdaSearch

def allowed_file(filename):
    """Check whether the filename passed has one of the allowed extensions

    Args:
        filename (str): Filepath to check.

    Returns:
        bool: Whether the filepath has one of the allowed extensions.
    """
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
        im = open_image(file.stream)
        data = BytesIO()
        im.save(data, "JPEG")
        enc_img = base64.b64encode(data.getvalue())
        dec_img = enc_img.decode('utf-8')
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')

        if app.config['method'] == 'local':
            engine = Resnet50Search()
            im = enc_img
        else:
            engine = LambdaSearch()
            im = dec_img

        return render_template('index.html', img_data=dec_img, images=engine.infer(im, k))
        # return render_template('index.html', img_data=dec_img, images=infer(enc_img, k))
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

if __name__ == "__main__":
    # RUNMETHOD=local;flask run
    app.run()
