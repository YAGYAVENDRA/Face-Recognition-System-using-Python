import os
import json
import base64
import cv2
from flask import Flask, render_template, request, jsonify, redirect, url_for
import face_recognition
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Ensure required directories exist
os.makedirs('static/images', exist_ok=True)

# Path to store user data
USER_DATA_FILE = 'user_data.json'

# Initialize user data file if it doesn't exist
if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump([], f)

def load_user_data():
    with open(USER_DATA_FILE, 'r') as f:
        return json.load(f)

def save_user_data(data):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image with improved error handling"""
    try:
        # Make sure we're getting the data part of the base64 string
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 to binary data
        img_data = base64.b64decode(base64_string)
        
        # Convert binary data to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        
        # Decode the numpy array as an image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image data")
            
        return img
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

def get_face_encoding(image):
    """Get face encoding with improved error handling and color conversion"""
    if image is None:
        return None
        
    # Convert BGR to RGB (face_recognition expects RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Check if conversion was successful
    if rgb_image is None:
        print("Color conversion failed")
        return None
    
    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_image)
    
    if not face_locations:
        print("No faces detected in the image")
        return None
    
    try:
        # Get encoding of the first face found
        face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
        return face_encoding.tolist()
    except Exception as e:
        print(f"Error encoding face: {str(e)}")
        return None

def save_image_file(image, filename):
    """Save image with proper color handling"""
    try:
        # Ensure the directory exists
        os.makedirs('static/images', exist_ok=True)
        
        # Save the image
        success = cv2.imwrite(filename, image)
        
        if not success:
            raise ValueError("Failed to save image file")
            
        return True
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return False

def find_user_by_face(face_encoding):
    users = load_user_data()
    if not users:
        return None
    
    face_encoding = np.array(face_encoding)
    
    for user in users:
        stored_encoding = np.array(user['face_encoding'])
        # Compare faces with a tolerance
        distance = face_recognition.face_distance([stored_encoding], face_encoding)[0]
        if distance < 0.6:  # Adjust tolerance as needed
            return user
    
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/verify')
def verify():
    return render_template('verify.html')

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    name = data.get('name')
    image_data = data.get('image')
    
    if not name or not image_data:
        return jsonify({'success': False, 'message': 'Name and image are required'}), 400
    
    try:
        # Convert base64 to image
        image = base64_to_image(image_data)
        
        # Get face encoding
        face_encoding = get_face_encoding(image)
        if not face_encoding:
            return jsonify({'success': False, 'message': 'No face detected in the image. Please try again with better lighting.'}), 400
        
        # Save image file
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        image_filename = f"{name.replace(' ', '_')}_{timestamp}.jpg"
        image_path = os.path.join('static/images', image_filename)
        
        if not save_image_file(image, image_path):
            return jsonify({'success': False, 'message': 'Failed to save image. Please try again.'}), 500
        
        # Save user data
        users = load_user_data()
        user_data = {
            'id': len(users) + 1,
            'name': name,
            'face_encoding': face_encoding,
            'image_path': image_path,
            'registered_at': datetime.now().isoformat()
        }
        
        users.append(user_data)
        save_user_data(users)
        
        return jsonify({
            'success': True, 
            'message': f'User {name} registered successfully!',
            'user': {
                'id': user_data['id'],
                'name': user_data['name'],
                'image_path': user_data['image_path']
            }
        })
    except Exception as e:
        print(f"Registration error: {str(e)}")
        return jsonify({'success': False, 'message': f'Error during registration: {str(e)}'}), 500

@app.route('/api/verify', methods=['POST'])
def api_verify():
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False, 'message': 'Image is required'}), 400
    
    try:
        # Convert base64 to image
        image = base64_to_image(image_data)
        
        # Get face encoding
        face_encoding = get_face_encoding(image)
        if not face_encoding:
            return jsonify({'success': False, 'message': 'No face detected in the image. Please try again with better lighting.'}), 400
        
        # Find matching user
        user = find_user_by_face(face_encoding)
        
        if user:
            return jsonify({
                'success': True,
                'message': f'Welcome back, {user["name"]}!',
                'user': {
                    'id': user['id'],
                    'name': user['name'],
                    'image_path': user['image_path']
                }
            })
        else:
            return jsonify({'success': False, 'message': 'User not recognized. Please register first.'}), 404
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return jsonify({'success': False, 'message': f'Error during verification: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)