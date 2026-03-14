import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import MeanSquaredError
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

# Define the custom layer to load the model properly
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1.0, 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(var + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config

# Constants
IMG_SIZE = 256
CHANNELS = 3
REAL_IMAGE_PATH = './content/train/photos'

# Load both generator and discriminator models
def load_models(generator_path, discriminator_path):
    custom_objects = {'InstanceNormalization': InstanceNormalization}
    generator = load_model(generator_path, custom_objects=custom_objects)
    discriminator = load_model(discriminator_path, custom_objects=custom_objects)
    return generator, discriminator

# Preprocess sketch images
def preprocess_sketch(sketch_path, size=(IMG_SIZE, IMG_SIZE)):
    sketch_img = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
    sketch_img = cv2.resize(sketch_img, size)
    sketch_img = np.stack([sketch_img] * CHANNELS, axis=-1)
    sketch_img = (sketch_img.astype(np.float32) / 127.5) - 1
    return np.expand_dims(sketch_img, axis=0)  # Add batch dimension

# Generate a face from a sketch with quality assessment
def generate_face(generator, discriminator, sketch_path, output_path=None):
    # Get a random real image for reference
    real_image_paths = [os.path.join(REAL_IMAGE_PATH, f) for f in os.listdir(REAL_IMAGE_PATH) if f.endswith(('.jpg', '.png'))]
    real_image_path = random.choice(real_image_paths)
    real_img = cv2.imread(real_image_path)
    real_img = cv2.resize(real_img, (IMG_SIZE, IMG_SIZE))
    real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
    real_img = (real_img.astype(np.float32) / 127.5) - 1
    
    # Preprocess the sketch
    preprocessed_sketch = preprocess_sketch(sketch_path)
    
    # Generate multiple faces and select the best one based on discriminator score
    num_attempts = 3
    best_face = None
    best_score = -float('inf')
    
    for _ in range(num_attempts):
        generated_face = generator(preprocessed_sketch, training=False)
        quality_score = discriminator(generated_face, training=False)
        
        if quality_score.numpy().mean() > best_score:
            best_score = quality_score.numpy().mean()
            best_face = generated_face
    
    generated_face = best_face.numpy()
    
    # Convert from [-1, 1] to [0, 1] range
    generated_face = (generated_face[0] * 0.5 + 0.5).astype(np.float32)
    generated_face = np.clip(generated_face, 0, 1)
    
    # Display the results
    plt.figure(figsize=(15, 5))
    
    # Display the sketch
    plt.subplot(1, 3, 1)
    sketch_display = (preprocessed_sketch[0] * 0.5 + 0.5).astype(np.float32)
    plt.imshow(sketch_display[:, :, 0], cmap='gray')
    plt.title('Input Sketch', pad=20)
    plt.axis('off')
    
    # Display the generated face
    plt.subplot(1, 3, 2)
    plt.imshow(generated_face)
    plt.title(f'Generated Face\nQuality Score: {best_score:.2f}', pad=20)
    plt.axis('off')
    
    # Display the real reference image
    plt.subplot(1, 3, 3)
    real_img_display = (real_img * 0.5 + 0.5).astype(np.float32)
    real_img_display = np.clip(real_img_display, 0, 1)
    plt.imshow(real_img_display)
    plt.title('Real (Reference)', pad=20)
    plt.axis('off')
    
    plt.tight_layout(pad=2.0)
    
    # Save the result if output path is provided
    if output_path:
        # Save the figure
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5)
        
        # Also save just the generated face as a separate image
        face_output_path = output_path.replace('.png', '_face.png')
        plt.figure()
        plt.imshow(generated_face)
        plt.axis('off')
        plt.savefig(face_output_path, bbox_inches='tight')
        plt.close('all')  # Close all figures to prevent memory leaks
    
    return generated_face

# Main function
def main(sketch_path):
    # Paths to the latest generator and discriminator checkpoints
    generator_path = 'checkpoints/generator_epoch_1000 (5).h5'
    discriminator_path = 'checkpoints/discriminator_epoch_1000 (5).h5'
    
    # Check if the checkpoints exist
    if not os.path.exists(generator_path) or not os.path.exists(discriminator_path):
        print(f"Error: Checkpoints not found")
        return
    
    # Load both models
    print("Loading the models...")
    generator, discriminator = load_models(generator_path, discriminator_path)
    
    # Directory to save results
    os.makedirs('results', exist_ok=True)
    
    if not os.path.exists(sketch_path):
        print(f"Error: Sketch not found at {sketch_path}")
        return
    
    # Generate face from sketch
    print("Generating face from sketch...")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"static/results/{current_time}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"generated_{os.path.basename(sketch_path).split('.')[0]}.png")
    generate_face(generator, discriminator, sketch_path, output_path)
    output_path = os.path.join(output_dir, f"generated_{os.path.basename(sketch_path).split('.')[0]}_face.png")
    print(f"Generated face saved to {output_path}")
    return output_path

from flask import Flask, request, render_template
from new import generate_image_from_sketch

app = Flask(__name__)

@app.route('/') #root
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            # Check if the uploads directory exists
            uploads_dir = 'uploads'
            if not os.path.exists(uploads_dir):
                os.makedirs(uploads_dir)
            # Save the uploaded image to the uploads directory with a unique name
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(uploads_dir, f"{current_time}_{image.filename}")
            image.save(image_path)
            result_img_path = main(image_path)
            # result_img_path = generate_image_from_sketch(image_path, True)  # This Runs your function !!!!!!!!!!!
            result_img_path = f"/{result_img_path}"  # Ensure it's a relative path
            print("*********", result_img_path)
            return { 'status': 'success', 'result_img_path': result_img_path }
        else:
            return { 'status': 'error', 'message': 'No image uploaded' }
    return { 'status': 'error', 'message': 'Invalid request method' }

if __name__ == "__main__":
    # index("./content/test/mysket/82.jpg")
    app.run(host='0.0.0.0', debug=True, port=9898) #http://{ipaddress}:{portnumber}
    
#before this create a virtual environment using command python -m venv venv
#activate the virtual environment using command source venv/bin/activate
#install the required packages using command pip install -r requirements.txt
# Need to run this function by using command python app.py