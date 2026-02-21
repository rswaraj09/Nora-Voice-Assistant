import cv2
import os
import sys

print("="*70)
print("FACE SAMPLE CAPTURE PROGRAM")
print("="*70)

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# Initialize camera
print("\nInitializing camera...")
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam.isOpened():
    print("❌ ERROR: Failed to open camera. Check if camera is connected and not in use.")
    sys.exit(1)

print("✓ Camera initialized successfully")

# Set camera properties
cam.set(3, 640)  # Frame width
cam.set(4, 480)  # Frame height

# Load Haar Cascade classifier
cascade_path = os.path.join(script_dir, 'haarcascade_frontalface_default.xml')
print(f"Loading classifier from: {cascade_path}")

detector = cv2.CascadeClassifier(cascade_path)

if detector.empty():
    print(f"❌ ERROR: Could not load classifier from {cascade_path}")
    cam.release()
    sys.exit(1)

print("✓ Classifier loaded successfully")

# Check write permissions
test_write_path = os.path.join(script_dir, '.write_test')
try:
    with open(test_write_path, 'w') as f:
        f.write('test')
    os.remove(test_write_path)
    print("✓ Directory is writable")
except Exception as e:
    print(f"❌ ERROR: Cannot write to directory: {e}")
    cam.release()
    sys.exit(1)

# Create samples directory
samples_dir = os.path.join(script_dir, 'samples')
os.makedirs(samples_dir, exist_ok=True)
print(f"✓ Samples directory ready: {samples_dir}")

print("\n" + "-"*70)
face_id = input("Enter a Numeric user ID here: ")
print("-"*70)
print("Starting face capture...")
print("  - Position your face in front of the camera")
print("  - Press 'ESC' to stop")
print("  - Need 100 samples for good training")
print("-"*70 + "\n")

count = 0
frame_count = 0
face_found_frames = 0

try:
    while True:
        ret, frame = cam.read()
        
        if not ret:
            print("❌ ERROR: Failed to read frame from camera")
            break
        
        frame_count += 1
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces (lowered threshold for better detection)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))
        
        # Prepare display frame
        display_frame = frame.copy()
        
        # Add info text
        info_text = f"Frame: {frame_count} | Faces: {len(faces)} | Samples: {count}/100"
        cv2.putText(display_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Process detected faces
        if len(faces) > 0:
            face_found_frames += 1
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                count += 1
                
                # Extract and save face
                face_roi = gray[y:y+h, x:x+w]
                
                # Create filename
                filename = f"face.{face_id}.{count}.jpg"
                filepath = os.path.join(samples_dir, filename)
                
                # Save the image
                success = cv2.imwrite(filepath, face_roi)
                
                if success:
                    print(f"✓ Sample {count:3d} saved: {filename}")
                    sys.stdout.flush()
                else:
                    print(f"❌ Sample {count} FAILED to save: {filepath}")
                    sys.stdout.flush()
        
        # Always show the camera window
        cv2.imshow('Face Capture - Press ESC to exit', display_frame)
        
        # Check for key press
        key = cv2.waitKey(100) & 0xff
        if key == 27:  # ESC key
            print("\n⚠ ESC pressed - Stopping capture")
            break
        elif count >= 100:
            print("\n✓ 100 samples reached!")
            break

except Exception as e:
    print(f"\n❌ ERROR during capture: {e}")

# Cleanup
print("\n" + "="*70)
print("CAPTURE SUMMARY")
print("="*70)
print(f"Total frames processed: {frame_count}")
print(f"Frames with faces detected: {face_found_frames}")
print(f"Total samples saved: {count}")
print(f"Save location: {samples_dir}")

# List saved files
saved_files = [f for f in os.listdir(samples_dir) if f.startswith(f"face.{face_id}")]
print(f"Files in folder: {len(saved_files)}")
if saved_files:
    print("\nSaved files:")
    for f in sorted(saved_files)[:5]:
        print(f"  - {f}")
    if len(saved_files) > 5:
        print(f"  ... and {len(saved_files) - 5} more")

print("="*70)
print("Closing camera and windows...")
cam.release()
cv2.destroyAllWindows()
print("✓ Program finished")
