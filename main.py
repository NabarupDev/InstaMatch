import cv2
import os
import numpy as np
import math
import csv
import datetime
import time
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv

def load_known_faces(faces_dir):
    """Load known faces from directory using OpenCV's face recognizer"""
    if not os.path.exists(faces_dir):
        print(f"Directory '{faces_dir}' not found. Creating it...")
        os.makedirs(faces_dir)
        print(f"Please add face images to the '{faces_dir}' directory and run the program again.")
        print("Each image should contain one clear face and be named with the person's name (e.g., 'john.jpg').")
        return None, {}
        
    if len([f for f in os.listdir(faces_dir) if f.endswith('.jpg') or f.endswith('.png')]) == 0:
        print(f"No images found in '{faces_dir}' directory.")
        print("Please add face images to the directory and run the program again.")
        print("Each image should contain one clear face and be named with the person's name (e.g., 'john.jpg').")
        return None, {}
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2,       
        neighbors=8,     
        grid_x=8,        
        grid_y=8,
        threshold=250    
    )
    
    faces = []
    labels = []
    label_ids = {}
    current_id = 0
    
    print("Starting to load and process known faces...")
    
    for filename in os.listdir(faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name = os.path.splitext(filename)[0]
            
            # Assign a numeric ID to each name
            if name not in label_ids:
                label_ids[name] = current_id
                current_id += 1
            
            # Load image file
            image_path = os.path.join(faces_dir, filename)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image {filename}. Skipping...")
                continue
            
            print(f"Processing image: {filename}")    
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization to improve contrast
            gray = cv2.equalizeHist(gray)
            
            # Try to detect faces from multiple angles
            faces_rect = detect_face_multi_angle(gray, face_cascade, profile_cascade)
            
            # Manual override for training - use the whole image if no face detected
            if len(faces_rect) == 0:
                print(f"No face detected in {filename} - using entire image.")
                h, w = gray.shape
                faces_rect = np.array([[0, 0, w, h]])
            
            # Process the detected face
            for (x, y, w, h) in faces_rect:
                face_roi = gray[y:y+h, x:x+w]
                
                # Resize to standard size
                face_roi = cv2.resize(face_roi, (100, 100))
                
                # Apply further preprocessing 
                # Normalize the image
                face_roi = cv2.equalizeHist(face_roi)
                
                # Add the processed face to our training data
                faces.append(face_roi)
                labels.append(label_ids[name])
                
                # Create variations with enhanced angle augmentation
                augmented_faces = create_angle_variations(face_roi)
                for aug_face in augmented_faces:
                    faces.append(aug_face)
                    labels.append(label_ids[name])
                
                print(f"Added face: {name} with angle variations for training")
    
    # Create a reverse mapping from ID to name
    id_to_name = {v: k for k, v in label_ids.items()}
    
    # Debug information
    print(f"Total faces for training: {len(faces)}")
    print(f"Label IDs: {label_ids}")
    
    # Train recognizer if we have data
    if faces:
        print("Training face recognizer...")
        recognizer.train(faces, np.array(labels))
        print("Training complete.")
        return recognizer, id_to_name
    else:
        print("No valid face images found. Please add clear face images to the directory.")
        return None, {}

def create_angle_variations(face_img):
    """Create variations of the face at different angles"""
    variations = []
    
    # Brightness variations
    bright = cv2.convertScaleAbs(face_img, alpha=1.2, beta=10)
    dark = cv2.convertScaleAbs(face_img, alpha=0.8, beta=-10)
    variations.extend([bright, dark])
    
    # Rotation variations for simulating different viewing angles
    angles = [-15, -10, -5, 5, 10, 15]  # More angles for better coverage
    center = (face_img.shape[1] // 2, face_img.shape[0] // 2)
    
    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))
        variations.append(rotated)
        
        # Also add brightness variations to rotated images for more robustness
        if angle % 10 == 0:  # Only for some angles to avoid too many images
            bright_rotated = cv2.convertScaleAbs(rotated, alpha=1.2, beta=10)
            variations.append(bright_rotated)
    
    # Add horizontal flip to simulate profile from other side
    flipped = cv2.flip(face_img, 1)
    variations.append(flipped)
    
    # Add slight perspective transforms to simulate head tilts
    h, w = face_img.shape
    src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    
    # Slight perspective variations
    dst_variations = [
        # Tilt right
        np.float32([[0, 10], [w-1, 0], [0, h-11], [w-1, h-1]]),
        # Tilt left
        np.float32([[0, 0], [w-1, 10], [0, h-1], [w-1, h-11]]),
        # Looking up slightly
        np.float32([[5, 0], [w-6, 0], [0, h-1], [w-1, h-1]]),
        # Looking down slightly
        np.float32([[0, 0], [w-1, 0], [5, h-1], [w-6, h-1]])
    ]
    
    for dst_points in dst_variations:
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(face_img, M, (w, h))
        variations.append(warped)
    
    return variations

def detect_face_multi_angle(gray_img, frontal_cascade, profile_cascade):
    """Detect faces from multiple angles including profiles"""
    faces = []
    
    # Try to detect frontal face first (most common)
    frontal_faces = frontal_cascade.detectMultiScale(
        gray_img, 
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )
    
    if len(frontal_faces) > 0:
        return frontal_faces
    
    # Try to detect profile faces
    profile_faces = profile_cascade.detectMultiScale(
        gray_img, 
        scaleFactor=1.1,
        minNeighbors=3,  # Lower threshold for profile detection
        minSize=(30, 30)
    )
    
    if len(profile_faces) > 0:
        return profile_faces
    
    # Try profile faces from the other side (flip the image)
    flipped = cv2.flip(gray_img, 1)
    profile_faces_flipped = profile_cascade.detectMultiScale(
        flipped, 
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30)
    )
    
    # Adjust coordinates for flipped faces
    if len(profile_faces_flipped) > 0:
        w_img = gray_img.shape[1]
        for i, (x, y, w, h) in enumerate(profile_faces_flipped):
            # Adjust x coordinate for the flip
            profile_faces_flipped[i][0] = w_img - x - w
        return profile_faces_flipped
    
    # Try with different parameters as a last resort
    for scale in [1.05, 1.2]:
        for min_neighbors in [2, 3, 5]:
            frontal_attempt = frontal_cascade.detectMultiScale(
                gray_img, 
                scaleFactor=scale,
                minNeighbors=min_neighbors,
                minSize=(30, 30)
            )
            if len(frontal_attempt) > 0:
                return frontal_attempt
    
    return []

def setup_attendance_system():
    """Set up the attendance system directories and files"""
    # Create attendance directory if it doesn't exist
    attendance_dir = "attendance"
    if not os.path.exists(attendance_dir):
        os.makedirs(attendance_dir)
    
    # Get current date for the attendance file
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    attendance_file = os.path.join(attendance_dir, f"attendance_{current_date}.csv")
    
    # Create the CSV file with headers if it doesn't exist
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Time", "Date"])
    
    # Connect to MongoDB
    try:
        # Load environment variables from .env file
        load_dotenv()
        
        # Get MongoDB connection string from environment variable
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        
        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        
        # Access/create the attendance database
        db = client["attendance_db"]
        
        # Access/create the attendance collection
        collection = db["attendance_records"]
        
        # Create an index on name and date for faster queries
        collection.create_index([("name", pymongo.ASCENDING), ("timestamp", pymongo.ASCENDING)])
        
        print("Connected to MongoDB successfully")
        return attendance_file, collection
    
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        return attendance_file, None

def check_attendance(name, attendance_file, mongo_collection=None):
    """Check if a person's attendance has already been marked today"""
    # Get current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # First check MongoDB if available
    if mongo_collection:
        try:
            # Create date range for today (start of day to end of day)
            start_date = datetime.datetime.strptime(f"{current_date}T00:00:00", "%Y-%m-%dT%H:%M:%S")
            end_date = datetime.datetime.strptime(f"{current_date}T23:59:59", "%Y-%m-%dT%H:%M:%S")
            
            # Query MongoDB for attendance record of this person today
            result = mongo_collection.find_one({
                "name": name,
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            })
            
            if result:
                print(f"Found MongoDB record for {name} today")
                return True
        except Exception as e:
            print(f"Error querying MongoDB: {e}")
    
    # Fall back to checking CSV file
    if not os.path.exists(attendance_file):
        return False
    
    with open(attendance_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            if row and row[0] == name:
                return True
    
    return False

def mark_attendance(name, attendance_file, confidence=0.9, mongo_collection=None, camera_id="CAM-001"):
    """Mark a person's attendance in the attendance file and MongoDB"""
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y-%m-%d")
    
    # Write to CSV file
    with open(attendance_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, current_time, current_date])
    
    # Write to MongoDB if available
    if mongo_collection:
        try:
            # Format timestamp as ISO format
            timestamp = now.isoformat()
            
            # Insert attendance record into MongoDB
            record = {
                "name": name,
                "timestamp": now,  # Store as datetime object for better querying
                "camera_id": camera_id,
                "confidence": confidence
            }
            
            result = mongo_collection.insert_one(record)
            print(f"MongoDB record created with ID: {result.inserted_id}")
        except Exception as e:
            print(f"Error inserting into MongoDB: {e}")
    
    print(f"Marked attendance for {name} at {current_time}")
    return True

def main():
    """Main function to run the attendance system"""
    # Create a .env file if it doesn't exist (with default MongoDB URL)
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("MONGO_URI=mongodb://localhost:27017/")
        print("Created .env file with default MongoDB connection string")
    
    # Rest of the main function
    # Set the path to the known faces directory
    known_faces_dir = "known_faces"
    
    # Set up attendance system
    attendance_file = setup_attendance_system()
    
    # Load face detectors for multiple angles
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    # Load known faces and train recognizer
    print("Loading known faces...")
    recognizer, id_to_name = load_known_faces(known_faces_dir)
    
    # Check if we have face data
    if recognizer is None or len(id_to_name) == 0:
        print("No face data available. Exiting program.")
        return
    
    print(f"Loaded {len(id_to_name)} known faces")
    
    # Initialize webcam with low resolution for better performance
    print("Starting webcam...")
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Higher resolution for better recognition
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not video_capture.isOpened():
        print("Could not open webcam")
        return

    print("Processing video... Press 'q' to quit")
    
    # Lower threshold for recognition with angled faces
    recognition_threshold = 120  # Even more permissive
    
    # Dictionary to track attendance status and cooldown
    recognized_persons = {}
    cooldown_period = 5  # seconds between status updates
    
    # Dictionary to track which persons have had attendance marked
    attendance_marked = {}
    
    # For stability of recognition
    last_name = None
    name_stability_counter = 0
    
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)
        
        # Try multi-angle face detection
        faces = detect_face_multi_angle(gray, face_cascade, profile_cascade)
        
        # Current time for cooldown checks
        current_time = time.time()
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Apply preprocessing
            face_roi = cv2.equalizeHist(face_roi)
            face_resized = cv2.resize(face_roi, (100, 100))
            
            # Recognize face with enhanced angle tolerance
            try:
                # Attempt recognition with the direct face
                direct_confidence = float('inf')
                try:
                    label_id, direct_confidence = recognizer.predict(face_resized)
                except:
                    pass
                
                # Also try with slight rotations for better angle matching
                best_confidence = direct_confidence
                best_label = label_id if direct_confidence < float('inf') else -1
                
                for angle in [-8, -4, 0, 4, 8]:  # Try slight rotations
                    if angle == 0:
                        continue  # Already tried with direct face
                        
                    center = (face_resized.shape[1] // 2, face_resized.shape[0] // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(face_resized, M, (face_resized.shape[1], face_resized.shape[0]))
                    
                    try:
                        rot_label, rot_confidence = recognizer.predict(rotated)
                        if rot_confidence < best_confidence:
                            best_confidence = rot_confidence
                            best_label = rot_label
                    except:
                        continue
                
                # Use the best result found
                label_id, confidence = best_label, best_confidence
                
                # Debug information
                print(f"Best recognition - Label: {label_id}, Confidence: {confidence}")
                
                # Handle recognition results with more permissive threshold
                if confidence < recognition_threshold:  # Good match with higher threshold
                    name = id_to_name[label_id]
                    
                    # Stability: if same name repeatedly detected, increase confidence
                    if name == last_name:
                        name_stability_counter += 1
                    else:
                        name_stability_counter = 0
                        last_name = name
                    
                    # Status text to display
                    status_text = ""
                    
                    # If stable recognition (at least 3 consecutive frames)
                    if name_stability_counter > 2:
                        # Check if attendance needs to be marked with cooldown
                        if name not in recognized_persons or (current_time - recognized_persons[name]['timestamp'] > cooldown_period):
                            # Check if attendance is already marked for today
                            if name in attendance_marked and attendance_marked[name]:
                                status_text = f"Already marked"
                            else:
                                # Check attendance in CSV file
                                if check_attendance(name, attendance_file):
                                    attendance_marked[name] = True
                                    status_text = f"Already marked"
                                else:
                                    # Mark attendance
                                    if mark_attendance(name, attendance_file):
                                        attendance_marked[name] = True
                                        status_text = f"Marked at {datetime.datetime.now().strftime('%H:%M:%S')}"
                            
                            # Update recognition timestamp
                            recognized_persons[name] = {
                                'timestamp': current_time,
                                'status': status_text
                            }
                        else:
                            # Use stored status during cooldown period
                            status_text = recognized_persons[name]['status']
                        
                        # Display name and status
                        result_text = f"{name} - {status_text}"
                    else:
                        # Not stable enough for attendance yet, just show name
                        match_percent = max(0, min(100, int(100 - confidence/2)))
                        result_text = f"{name} ({match_percent}%)"
                else:
                    # Poor match, show as unknown
                    name = "Unknown"
                    last_name = None
                    name_stability_counter = 0
                    
                    # Calculate nearest match info for debugging
                    nearest_match = id_to_name[label_id] if label_id in id_to_name else "None"
                    print(f"Unknown face, closest match was {nearest_match} with confidence {confidence}")
                    
                    result_text = f"Unknown ({int(min(100, confidence/2))}%)"
                
            except Exception as e:
                # More detailed exception handling
                print(f"Error in face recognition: {e}")
                result_text = "Unknown"
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Draw name label with background
            # Ensure the background doesn't go out of frame bounds
            label_height = 25
            # Check if label would extend beyond the bottom of the frame
            max_available_height = min(label_height, frame.shape[0] - (y+h))
            
            if max_available_height > 0:  # Only if we have space for the label
                label_background = np.zeros((max_available_height, w, 3), dtype=np.uint8)
                label_background[:] = (0, 0, 255)  # Red background
                
                # Safely overlay the background on the frame
                frame[y+h:y+h+max_available_height, x:x+w] = label_background
                
                # Put the text on the background if there's enough room
                if max_available_height >= 15:  # Minimum height for text
                    cv2.putText(frame, result_text, (x+5, y+h+17), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            else:
                # Fall back to just text without background if no space
                cv2.putText(frame, result_text, (x+5, y+h), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        
        # Add attendance system info on top of the frame
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Attendance System - {current_datetime}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        
        cv2.imshow('Attendance System', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()