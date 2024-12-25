import cv2
from keras.models import model_from_json
import numpy as np

# Load the trained model from JSON file and weights
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load the face cascade classifier
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from the face image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Open the webcam for capturing video
webcam = cv2.VideoCapture(0)

# Define emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Main loop to capture video and perform face detection
while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        for (x, y, w, h) in faces:
            # Extract face region and preprocess for prediction
            face_image = gray[y:y+h, x:x+w]
            face_image = cv2.resize(face_image, (48, 48))
            img = extract_features(face_image)

            # Make prediction using the loaded model
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            # Draw rectangle around the face and label the emotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, prediction_label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the annotated frame
        cv2.imshow("Emotion Detection", frame)

        # Check for 'ESC' key press to exit
        if cv2.waitKey(1) == 27:  # ESC key
            break

    except cv2.error:
        pass

# Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()
