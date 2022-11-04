from __future__ import division, print_function

import os
import numpy as np


from keras.models import load_model
from keras import utils

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


print('Model loaded. Check http://127.0.0.1:5000/home')

l={0:"No disease",
   1:"Mild non-prolifirative diabetic retinopathy",
   2:"Moderare non-prolifirative diabetic retinopathy",
   3:"Severe non-prolifirative diabetic retinopathy",
   4:"Prolifirative diabetic retinopathy"
   }

def model_predict(img_path):
        model=load_model('diabetic.h5')
        img = utils.load_img(img_path, target_size=(299, 299))
        x = utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        preds=np.argmax(model.predict(x), axis=1)
        return preds
   
        

@app.route('/home', methods=['GET'])
def index():
    # Main page
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    print("hi")
    if request.method == 'POST':
        f = request.files['fileq']
       

      
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        
        preds = model_predict(file_path)

       
        result = str(preds)
        print(result)          
        
        return render_template('homer.html',data=result)
    return None


if __name__ == '__main__':
    app.run(debug=True)
    