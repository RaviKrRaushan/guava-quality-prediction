from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import tempfile
import uuid
import base64
import json

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_flash_messages')  # For flash messages
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Email Configuration - IMPORTANT: Replace with your actual credentials or environment variables
# For production, use environment variables to store sensitive info like passwords.
# Example: app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_USERNAME'] = 'ivar.kr000@gmail.com'  # Replace with your Gmail
app.config['MAIL_PASSWORD'] = 'mevx wgmy lvja klkv'  # Use an App Password, not your regular password

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'jpg', 'jpeg', 'png', 'gif', 'webm'}

# Load the trained MobileNetV2 model
MODEL_PATH = 'best_model_mobilenet.h5'
model = None # Initialize model to None
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    else:
        logger.error(f"Model file not found at: {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {e}")


# Classes for prediction
CLASSES = ['fresh', 'mid', 'rotten']

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_image_file(filename):
    """Check if file is an image based on extension"""
    image_extensions = {'jpg', 'jpeg', 'png', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in image_extensions

def is_video_file(filename):
    """Check if file is a video based on extension"""
    video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions

def decode_base64_image(base64_string):
    """Decode base64 image data to numpy array"""
    try:
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode image using OpenCV
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_array is None:
            logger.error("cv2.imdecode failed, image array is None.")
            return None
        
        # Convert BGR to RGB (OpenCV reads in BGR by default)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        return img_array
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None

# Function to predict image quality
def predict_image(image_path_or_array, is_array=False):
    """Predict the quality of guava from image"""
    if model is None:
        logger.error("Model is not loaded. Cannot perform image prediction.")
        return {"success": False, "error": "Model not loaded properly. Please ensure 'best_model_mobilenet.h5' is in the correct directory."}
    
    try:
        if is_array:
            # Image is already a numpy array (from camera capture)
            img = image_path_or_array
        else:
            # Image is a file path (from file upload)
            img = cv2.imread(image_path_or_array)
            if img is None:
                logger.error(f"Could not read image file: {image_path_or_array}. File may be corrupted or unsupported.")
                return {"success": False, "error": "Could not read image file. The file may be corrupted or in an unsupported format."}
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        
        # Resize and normalize
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0) # Add batch dimension

        prediction = model.predict(img, verbose=0) # Added verbose=0 to suppress output
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = CLASSES[predicted_class_idx]
        confidence = float(prediction[0][predicted_class_idx])
        
        # Get all class probabilities
        class_probabilities = {cls: float(prediction[0][i]) for i, cls in enumerate(CLASSES)}

        # For single image, quality counts and percentages are straightforward
        quality_counts = {"Fresh": 0, "Mid": 0, "Rotten": 0}
        quality_counts[predicted_class.capitalize()] = 1

        quality_percentages = {"Fresh": 0.0, "Mid": 0.0, "Rotten": 0.0}
        quality_percentages[predicted_class.capitalize()] = 100.0
        
        return {
            "success": True,
            "prediction": predicted_class.capitalize(),
            "confidence": confidence,
            "file_type": "image",
            "frames_analyzed": 1,
            "predictions": class_probabilities, # Individual class probabilities for the image
            "overall_quality": predicted_class.capitalize(), # For single image, overall is its own prediction
            "quality_distribution": class_probabilities, # For single image, distribution is its own probabilities
            "quality_counts": quality_counts, # Added for consistency with video results
            "quality_percentages": quality_percentages # Added for consistency with video results
        }
    
    except Exception as e:
        logger.error(f"Error during image prediction: {e}")
        return {"success": False, "error": f"An error occurred during image prediction: {str(e)}"}

# Function to preprocess a frame for model prediction
def preprocess_frame(frame, target_size=(224, 224)):
    """Preprocess a frame for model prediction"""
    # Resize the frame
    resized = cv2.resize(frame, target_size)
    # Convert to RGB (our model was trained on RGB)
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize pixel values
    normalized = rgb_frame / 255.0
    # Expand dimensions to match model input shape (batch_size, height, width, channels)
    preprocessed = np.expand_dims(normalized, axis=0)
    return preprocessed

def frame_to_base64(frame):
    """Convert frame to base64 string for frontend display"""
    try:
        # Convert BGR to RGB (OpenCV reads in BGR, but usually display wants RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Encode as JPEG
        # Using a higher quality (e.g., 90) might result in larger data but better thumbnails
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90] 
        success, buffer = cv2.imencode('.jpg', rgb_frame, encode_param)
        
        if not success:
            logger.error("Failed to encode frame to JPEG buffer.")
            return None
            
        # Convert to base64 and add data URL prefix
        img_str = base64.b64encode(buffer).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error converting frame to base64: {e}")
        return None

# Function to extract frames from video with detailed information for frontend
def extract_frames_with_progress(video_path, max_frames=10):
    """Extract frames from video with detailed information for frontend"""
    cap = cv2.VideoCapture(video_path)
    frames_data = [] # To store actual frame numpy arrays
    frame_metadata = [] # To store metadata including thumbnails for frontend
    
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return [], []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        logger.warning(f"Total frames could not be determined for {video_path}. Reading up to {max_frames} frames sequentially.")
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames_data.append(frame)
            timestamp = frame_count / fps if fps > 0 else frame_count # Estimate timestamp
            frame_metadata.append({
                "frame_number": frame_count + 1,
                "timestamp": timestamp,
                "thumbnail": frame_to_base64(frame)
            })
            frame_count += 1
    else:
        # Extract evenly distributed frames
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            # Generate indices for evenly spaced frames
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames_data.append(frame)
                timestamp = frame_idx / fps if fps > 0 else 0 # Use 0 if fps is 0
                frame_metadata.append({
                    "frame_number": i + 1, # 1-indexed for display
                    "timestamp": timestamp,
                    "thumbnail": frame_to_base64(frame)
                })
            else:
                logger.warning(f"Could not read frame at index {frame_idx} from video {video_path}")
    
    cap.release()
    logger.info(f"Extracted {len(frames_data)} frames from video.")
    return frames_data, frame_metadata

# Function to predict video quality with detailed frame analysis
def predict_video_with_frames(video_path, max_frames=10):
    """Predict video quality with detailed frame-by-frame analysis"""
    if model is None:
        logger.error("Model is not loaded. Cannot perform video prediction.")
        return {"success": False, "error": "Model not loaded properly. Please ensure 'best_model_mobilenet.h5' is in the correct directory."}
    
    try:
        # Extract frames data and metadata
        frames_data, frame_metadata = extract_frames_with_progress(video_path, max_frames)
        
        if not frames_data:
            logger.error(f"No frames could be extracted from video: {video_path}")
            return {"success": False, "error": "No frames could be extracted from the video or video is unreadable."}
        
        # Process each frame
        frame_predictions_results = [] # Detailed predictions for each frame
        all_predictions_raw = [] # Raw prediction probabilities for overall average
        
        for i, frame in enumerate(frames_data):
            # Preprocess frame
            processed_frame = preprocess_frame(frame)
            
            # Make prediction
            pred = model.predict(processed_frame, verbose=0)[0] # Added verbose=0
            all_predictions_raw.append(pred)
            
            # Get frame prediction details
            frame_class_idx = np.argmax(pred)
            frame_class = CLASSES[frame_class_idx]
            frame_confidence = float(pred[frame_class_idx])
            
            frame_prediction_detail = {
                "frame_number": frame_metadata[i]["frame_number"],
                "prediction": frame_class.capitalize(),
                "confidence": frame_confidence,
                "predictions": {cls: float(pred[j]) for j, cls in enumerate(CLASSES)}, # All class probabilities for this frame
                "thumbnail": frame_metadata[i]["thumbnail"],
                "timestamp": frame_metadata[i]["timestamp"]
            }
            frame_predictions_results.append(frame_prediction_detail)
        
        # Calculate overall statistics
        avg_prediction = np.mean(all_predictions_raw, axis=0)
        predicted_class_idx = np.argmax(avg_prediction)
        predicted_class = CLASSES[predicted_class_idx]
        overall_confidence = float(avg_prediction[predicted_class_idx])
        
        # Quality distribution (average probabilities across all frames)
        quality_distribution = {cls: float(avg_prediction[i]) for i, cls in enumerate(CLASSES)}
        
        # Frame quality counts
        quality_counts = {"Fresh": 0, "Mid": 0, "Rotten": 0}
        for frame_pred in frame_predictions_results:
            quality_counts[frame_pred["prediction"]] += 1
        
        # Quality percentages
        total_frames_analyzed = len(frames_data)
        quality_percentages = {
            cls: (count / total_frames_analyzed) * 100 
            for cls, count in quality_counts.items()
        } if total_frames_analyzed > 0 else {"Fresh": 0.0, "Mid": 0.0, "Rotten": 0.0}

        result = {
            "success": True,
            "prediction": predicted_class.capitalize(), # Overall predicted class
            "confidence": overall_confidence, # Overall confidence
            "overall_quality": predicted_class.capitalize(), # Redundant, but kept for compatibility
            "frames_analyzed": total_frames_analyzed,
            "file_type": "video",
            "quality_distribution": quality_distribution,
            "frame_predictions": frame_predictions_results, # List of detailed frame predictions
            "frames": frame_predictions_results, # Alternative key for HTML compatibility
            "quality_counts": quality_counts,
            "quality_percentages": quality_percentages
        }
        
        # Debug logging
        logger.info(f"Video prediction result: frames_analyzed={total_frames_analyzed}, file_type=video")
        logger.info(f"Overall predicted quality: {result['prediction']} with confidence {result['confidence']:.2f}")
        logger.info(f"Frame predictions count: {len(frame_predictions_results)}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error during video prediction: {e}")
        return {"success": False, "error": f"An error occurred during video prediction: {str(e)}"}

def send_email(name, email, subject, message):
    """Send an email with the contact form data"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = app.config['MAIL_USERNAME']
        msg['To'] = app.config['MAIL_USERNAME']  # Sending to yourself
        msg['Subject'] = f"Guava Quality Predictor Contact: {subject}"
        
        # Email body
        body = f"""
        New contact form submission from Guava Quality Predictor:
        
        Name: {name}
        Email: {email}
        Subject: {subject}
        
        Message:
        {message}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls() # Enable TLS encryption
        server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        
        # Send email
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Contact email sent successfully from {email} with subject: {subject}")
        return True
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/submit-contact', methods=['POST'])
def submit_contact():
    """Handle contact form submission"""
    try:
        # Get form data
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject', 'No Subject')
        message = request.form.get('message')
        
        # Validate required fields
        if not all([name, email, message]):
            flash('Please fill in all required fields', 'danger')
            return redirect(url_for('contact'))
        
        # Send email
        email_sent = send_email(name, email, subject, message)
        
        if email_sent:
            flash('Your message has been sent successfully!', 'success')
        else:
            flash('There was an issue sending your message. Please try again later.', 'danger')
            
        return redirect(url_for('contact'))
        
    except Exception as e:
        logger.error(f"Error processing contact form: {e}")
        flash('An error occurred. Please try again later.', 'danger')
        return redirect(url_for('contact'))

@app.route('/check-camera')
def check_camera():
    """Simple endpoint to help debug camera access (not directly used by frontend for camera access)"""
    return jsonify({"status": "success", "message": "Camera check endpoint reached"})

@app.route('/predict', methods=['POST'])
def process_file():
    """Handle file upload (image/video) and camera capture (base64 image data) for prediction"""
    try:
        # Check for base64 image data (from camera capture) first, as it's JSON
        if request.is_json:
            data = request.get_json()
            if 'image_data' in data:
                logger.info("Received base64 image data from camera.")
                return handle_camera_capture(data['image_data'])
        
        # Check for regular file uploads (images or videos)
        elif 'file' in request.files: # Generic file input, check its type
            file = request.files['file']
            if file and file.filename and allowed_file(file.filename):
                if is_image_file(file.filename):
                    logger.info(f"Received uploaded image file: {file.filename}")
                    return handle_image_upload(file)
                elif is_video_file(file.filename):
                    logger.info(f"Received uploaded video file: {file.filename}")
                    return handle_video_upload(file)
                else:
                    return jsonify({"success": False, "error": "Unsupported file type provided in 'file' field"}), 400
        
        elif 'image' in request.files: # Specific image input field
            file = request.files['image']
            if file and file.filename and is_image_file(file.filename):
                logger.info(f"Received dedicated image file: {file.filename}")
                return handle_image_upload(file)
            else:
                return jsonify({"success": False, "error": "Invalid or unsupported image file provided in 'image' field"}), 400

        elif 'video' in request.files: # Specific video input field
            file = request.files['video']
            if file and file.filename and is_video_file(file.filename):
                logger.info(f"Received dedicated video file: {file.filename}")
                return handle_video_upload(file)
            else:
                return jsonify({"success": False, "error": "Invalid or unsupported video file provided in 'video' field"}), 400
        
        logger.warning("No valid file or image data found in request.")
        return jsonify({"success": False, "error": "No valid file or image data provided"}), 400
    
    except Exception as e:
        logger.error(f"Error in process_file: {e}")
        return jsonify({"success": False, "error": f"An unexpected error occurred: {str(e)}"}), 500

def handle_file_upload(file):
    """Deprecated/General handler, redirect to specific handlers based on type."""
    # This function is now mostly for fallback or if 'file' input is truly generic.
    # The 'process_file' route now handles dispatching more precisely.
    if is_image_file(file.filename):
        return handle_image_upload(file)
    elif is_video_file(file.filename):
        return handle_video_upload(file)
    else:
        logger.error(f"Unsupported file type in handle_file_upload: {file.filename}")
        return jsonify({"success": False, "error": "Unsupported file type"}), 400

def handle_video_upload(file):
    """Handle video file upload with frame extraction and prediction"""
    try:
        # Create a unique temporary directory for each upload
        temp_dir = tempfile.mkdtemp()
        filename = secure_filename(file.filename)
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        logger.info(f"Video saved temporarily to: {temp_path}")
        
        # Process video with detailed frame analysis
        result = predict_video_with_frames(temp_path)
        
        # Cleanup temporary file and directory
        os.remove(temp_path)
        os.rmdir(temp_dir)
        logger.info(f"Cleaned up temporary video file and directory: {temp_dir}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error handling video upload: {e}")
        # Ensure cleanup even if error occurs
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        return jsonify({"success": False, "error": f"An error occurred during video file processing: {str(e)}"}), 500

def handle_image_upload(file):
    """Handle image file upload"""
    try:
        # Create a unique temporary directory for each upload
        temp_dir = tempfile.mkdtemp()
        filename = secure_filename(file.filename)
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        logger.info(f"Image saved temporarily to: {temp_path}")
        
        # Process image
        result = predict_image(temp_path)
        
        # Cleanup temporary file and directory
        os.remove(temp_path)
        os.rmdir(temp_dir)
        logger.info(f"Cleaned up temporary image file and directory: {temp_dir}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error handling image upload: {e}")
        # Ensure cleanup even if error occurs
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        return jsonify({"success": False, "error": f"An error occurred during image file processing: {str(e)}"}), 500

def handle_camera_capture(image_data):
    """Handle camera capture (base64 image data)"""
    try:
        # Decode base64 image into a numpy array
        img_array = decode_base64_image(image_data)
        if img_array is None:
            logger.error("Failed to decode base64 image data for camera capture.")
            return jsonify({"success": False, "error": "Could not decode image data from camera."}), 400
        
        # Make prediction directly from the numpy array
        result = predict_image(img_array, is_array=True)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error handling camera capture: {e}")
        return jsonify({"success": False, "error": f"An error occurred during camera image processing: {str(e)}"}), 500

@app.route('/predict-frames', methods=['POST'])
def predict_frames_endpoint():
    """Handle frame-by-frame prediction for videos (allows specifying frame count)"""
    try:
        if 'video' not in request.files:
            return jsonify({"success": False, "error": "No video file provided"}), 400
        
        file = request.files['video']
        if not file or not file.filename or not is_video_file(file.filename):
            return jsonify({"success": False, "error": "Invalid video file"}), 400
        
        # Get number of frames to extract (default 10)
        frame_count = request.form.get('frame_count', 10)
        try:
            frame_count = min(max(int(frame_count), 1), 30)  # Limit between 1-30
        except ValueError:
            frame_count = 10
        
        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        filename = secure_filename(file.filename)
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        # Process video with specified frame count
        result = predict_video_with_frames(temp_path, frame_count)
        
        # Cleanup
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in predict_frames_endpoint: {e}")
        return jsonify({"success": False, "error": f"An error occurred in predict_frames endpoint: {str(e)}"}), 500

@app.route('/test-video-response')
def test_video_response():
    """Test endpoint to check video response structure"""
    mock_response = {
        "success": True,
        "prediction": "Fresh",
        "confidence": 0.95,
        "overall_quality": "Fresh",
        "frames_analyzed": 10,
        "file_type": "video",
        "quality_distribution": {"fresh": 0.7, "mid": 0.2, "rotten": 0.1},
        "frame_predictions": [
            {
                "frame_number": i+1,
                "prediction": "Fresh",
                "confidence": 0.9,
                "thumbnail": f"data:image/jpeg;base64,test_data_{i}",
                "timestamp": i * 0.5,
                "predictions": {"fresh": 0.9, "mid": 0.07, "rotten": 0.03}
            } for i in range(10)
        ],
        "frames": [  # Alternative key for HTML compatibility
            {
                "frame_number": i+1,
                "prediction": "Fresh",
                "confidence": 0.9,
                "thumbnail": f"data:image/jpeg;base64,test_data_{i}",
                "timestamp": i * 0.5
            } for i in range(10)
        ],
        "quality_counts": {"Fresh": 8, "Mid": 1, "Rotten": 1},
        "quality_percentages": {"Fresh": 80.0, "Mid": 10.0, "Rotten": 10.0}
    }
    return jsonify(mock_response)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "supported_formats": list(ALLOWED_EXTENSIONS)
    })

if __name__ == '__main__':
    # When running locally, Flask's debug mode provides auto-reloading and better error messages.
    # In a production environment, debug should be False.
    app.run(debug=True)
