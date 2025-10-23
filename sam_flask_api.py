from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import shutil
from PIL import Image
import sys
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set PYTHONPATH and environment
os.environ['PYTHONPATH'] = '/content/Futura_Face:' + os.environ.get('PYTHONPATH', '')
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5'  # For T4 GPU
sys.path.insert(0, '/content/SAM')

@app.route('/upload', methods=['POST'])
def upload_image():
    print("Received request:", request.form, request.files)  # Debug
    # Ensure directories exist
    os.makedirs('inference_results/input', exist_ok=True)
    os.makedirs('inference_results/30', exist_ok=True)
    os.makedirs('inference_results/50', exist_ok=True)
    os.makedirs('inference_coupled/30', exist_ok=True)
    os.makedirs('inference_coupled/50', exist_ok=True)

    # Get image and target age from request
    if 'image' not in request.files:
        # Fallback to UTKFace dataset
        try:
            df = pd.read_csv('utkface_full_preprocessed.csv')
            test_img_path = df['preprocessed_path'].iloc[0]
            shutil.copy(test_img_path, 'inference_results/input/uploaded_image.jpg')
            test_img_path = 'inference_results/input/uploaded_image.jpg'
        except Exception as e:
            print(f"Error accessing dataset: {str(e)}")
            return jsonify({'error': f'Error accessing dataset: {str(e)}'}), 500
    else:
        file = request.files['image']
        target_age = request.form.get('target_age', '30')
        test_img_path = 'inference_results/input/uploaded_image.jpg'
        file.save(test_img_path)

    # Preprocess image
    try:
        img = Image.open(test_img_path).convert('RGB')
        img = img.resize((512, 512))  # Updated to 512x512
        img.save(test_img_path)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

    # Run SAM inference
    cmd = f"""
    PYTHONPATH=/content/Futura_Face python scripts/inference.py \
      --exp_dir=pretrained_models \
      --checkpoint_path=pretrained_models/sam_ffhq_aging.pt \
      --data_path=inference_results/input \
      --test_batch_size=1 \
      --test_workers=1 \
      --couple_outputs \
      --target_age={target_age} \
      --resize_outputs
    """
    print(f"Running command: {cmd}")  # Debug
    result = os.system(cmd)
    if result != 0:
        print(f"Inference failed with code: {result}")
        return jsonify({'error': 'Inference failed'}), 500

    # Return the aged image
    aged_path = f'inference_results/{target_age}/uploaded_image.jpg'
    if not os.path.exists(aged_path):
        aged_path = f'inference_coupled/{target_age}/uploaded_image.jpg'
    if os.path.exists(aged_path):
        print(f"Sending image: {aged_path}")  # Debug
        return send_file(aged_path, mimetype='image/jpeg')
    print(f"No output image found at: {aged_path}")
    return jsonify({'error': f'No output image found for age {target_age}'}), 404

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
