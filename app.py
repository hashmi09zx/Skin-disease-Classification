from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='mobilenet_skin_model_final.tflite')
interpreter.allocate_tensors()

# Class names
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Mapping detailed information for each class
DISEASE_INFO = {
    'akiec': {
        'full_name': 'Actinic Keratoses and Intraepithelial Carcinoma',
        'description': 'A type of skin lesion that may become cancerous if untreated.',
        'prevention': 'Limit sun exposure, use sunscreen, wear protective clothing.',
        'steps': 'Consult a dermatologist, consider biopsy or removal.',
        'more_info': 'https://www.skincancer.org/skin-cancer-information/actinic-keratosis/'
    },
    'bcc': {
        'full_name': 'Basal Cell Carcinoma',
        'description': 'Most common type of skin cancer, often caused by sun exposure.',
        'prevention': 'Avoid tanning beds, use broad-spectrum sunscreen, regular skin checks.',
        'steps': 'Seek medical evaluation, treatments may include surgery or topical therapy.',
        'more_info': 'https://www.skincancer.org/skin-cancer-information/basal-cell-carcinoma/'
    },
    'bkl': {
        'full_name': 'Benign Keratosis-like Lesions',
        'description': 'Non-cancerous growths like seborrheic keratoses and solar lentigines.',
        'prevention': 'Sun protection, avoid skin trauma.',
        'steps': 'Usually no treatment needed, but suspicious growths should be checked.',
        'more_info': 'https://www.aad.org/public/diseases/a-z/seborrheic-keratoses-overview'
    },
    'df': {
        'full_name': 'Dermatofibroma',
        'description': 'Benign skin nodules often appearing on the lower legs.',
        'prevention': 'No known prevention methods.',
        'steps': 'Generally harmless, removal if symptomatic.',
        'more_info': 'https://www.dermnetnz.org/topics/dermatofibroma'
    },
    'mel': {
        'full_name': 'Melanoma',
        'description': 'Deadliest form of skin cancer originating in melanocytes.',
        'prevention': 'Avoid intense sun exposure, regular mole checks, sunscreen use.',
        'steps': 'Urgent medical care, surgery, possible immunotherapy or targeted therapy.',
        'more_info': 'https://www.cancer.org/cancer/melanoma-skin-cancer.html'
    },
    'nv': {
        'full_name': 'Melanocytic Nevi (Moles)',
        'description': 'Common benign skin growths, sometimes confused with melanoma.',
        'prevention': 'Sun protection to prevent changes, monitor moles for irregularities.',
        'steps': 'Regular monitoring, dermatological review if changes appear.',
        'more_info': 'https://www.aad.org/public/diseases/a-z/moles-overview'
    },
    'vasc': {
        'full_name': 'Vascular Lesions',
        'description': 'Abnormal growth of blood vessels, usually benign.',
        'prevention': 'No specific prevention.',
        'steps': 'Consult dermatologist if lesion changes or bleeds.',
        'more_info': 'https://www.ncbi.nlm.nih.gov/books/NBK532290/'
    }
}

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check allowed file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to prepare the image
def prepare_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.mobilenet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types are png, jpg, jpeg, gif'}), 400
    
    try:
        image_bytes = file.read()
        image = prepare_image(image_bytes)
        
        # Prepare input tensor for TFLite model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], image)
        
        # Run inference
        interpreter.invoke()

        # Get the output tensor
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = np.max(tf.nn.softmax(predictions))
        
        # Get Top-2 and Top-3 Predictions
        top_2 = np.argsort(predictions[0])[-2:][::-1]
        top_3 = np.argsort(predictions[0])[-3:][::-1]

        top_2_classes = [CLASS_NAMES[i] for i in top_2]
        top_3_classes = [CLASS_NAMES[i] for i in top_3]

        predicted_class = CLASS_NAMES[predicted_class_idx]
        info = DISEASE_INFO[predicted_class]
        
        return jsonify({
            'predicted_class': predicted_class,
            'full_name': info['full_name'],
            'description': info['description'],
            'prevention': info['prevention'],
            'steps': info['steps'],
            'more_info': info['more_info'],
            'confidence': float(confidence),
            'top_2_classes': top_2_classes,
            'top_3_classes': top_3_classes
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
