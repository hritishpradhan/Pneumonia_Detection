from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import io
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load your trained model
model = load_model(r'C:\Users\hpkvb\OneDrive\Desktop\Pneumonia_Project\chest_xray.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        # Read the image file into a BytesIO object
        img_bytes = io.BytesIO(file.read())
        img = image.load_img(img_bytes, target_size=(224, 224))
        
        # Convert the image to a numpy array
        x = image.img_to_array(img)
        
        # Log the shape and content of the image array
        logging.info(f'Image array shape: {x.shape}')
        logging.info(f'Image array content: {x}')
        
        # Expand dimensions to fit the model's input shape and preprocess
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Log the preprocessed input
        logging.info(f'Preprocessed input shape: {x.shape}')
        logging.info(f'Preprocessed input content: {x}')
        
        # Make prediction
        preds = model.predict(x)
        pred_class = np.argmax(preds, axis=1)
        
        # Log the prediction
        logging.info(f'Predictions: {preds}')
        logging.info(f'Predicted class: {pred_class}')
        
        prediction = 'Pneumonia' if pred_class[0] == 1 else 'Normal'
        
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
