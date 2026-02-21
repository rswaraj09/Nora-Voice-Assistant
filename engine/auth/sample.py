import cv2
import os
import sys

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) #create a video capture object which is helpful to capture videos through webcam

# Check if camera opened successfully
if not cam.isOpened():
    print("ERROR: Failed to open camera. Please check if your camera is connected and not in use.")
    exit()

cam.set(3, 640) # set video FrameWidth
cam.set(4, 480) # set video FrameHeight

# Load classifier with proper path handling
script_dir = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(script_dir, 'haarcascade_frontalface_default.xml')
detector = cv2.CascadeClassifier(cascade_path)

#Haar Cascade classifier is an effective object detection approach
if detector.empty():
    print(f"ERROR: Failed to load Haar Cascade classifier from {cascade_path}")
    print("Make sure 'haarcascade_frontalface_default.xml' exists in the engine/auth directory")
    cam.release()
    exit()

print(f"Script directory: {script_dir}")
print(f"Cascade loaded from: {cascade_path}")

face_id = input("Enter a Numeric user ID  here:  ")
#Use integer ID for every new face (0,1,2,3,4,5,6,7,8,9........)

print("Taking samples, look at camera ....... ")
count = 0 # Initializing sampling face count
face_detected_count = 0

# Ensure samples directory exists
samples_dir = os.path.join(script_dir, 'samples')
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)
    print(f"Created samples directory: {samples_dir}")
else:
    print(f"Using existing samples directory: {samples_dir}")

print(f"Full path where samples will be saved: {samples_dir}")
print("-" * 60)

while True:

    ret, img = cam.read() #read the frames using the above created object
    
    if not ret:
        print("ERROR: Failed to read frame from camera")
        break
        
    converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #The function converts an input image from one color space to another
    faces = detector.detectMultiScale(converted_image, 1.3, 5)
    
    # Display number of faces detected on window
    display_img = img.copy()
    cv2.putText(display_img, f"Faces detected: {len(faces)} | Samples: {count}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if len(faces) > 0:
        face_detected_count += 1

    for (x,y,w,h) in faces:

        cv2.rectangle(display_img, (x,y), (x+w,y+h), (255,0,0), 2) #used to draw a rectangle on any image
        count += 1

        sample_path = os.path.join(samples_dir, f"face.{face_id}.{count}.jpg")
        
        # Save the grayscale face image
        success = cv2.imwrite(sample_path, converted_image[y:y+h,x:x+w])
        
        # To capture & Save images into the datasets folder
        if success:
            print(f"✓ Sample {count} saved: {sample_path}")
            sys.stdout.flush()  # Force flush output
        else:
            print(f"✗ ERROR: Failed to save sample {count} to: {sample_path}")
            sys.stdout.flush()

    # Display the frame always (not just when face detected)
    cv2.imshow('Face Capture - Press ESC to exit', display_img) #Used to display an image in a window

    k = cv2.waitKey(100) & 0xff # Waits for a pressed key
    if k == 27: # Press 'ESC' to stop
        print("\nESC pressed - Stopping capture")
        break
    elif count >= 100: # Take 100 samples (More sample --> More accuracy)
         print(f"\nTarget reached - 100 samples captured")
         break

print(f"\n{'='*60}")
print(f"Total faces detected in stream: {face_detected_count}")
print(f"Total samples saved: {count}")
print(f"Samples location: {samples_dir}")
print(f"Samples taken now closing the program....")
print(f"{'='*60}")
cam.release()
cv2.destroyAllWindows()