import cv2
import tensorflow as tf
import numpy as np

# Define labels
labels = ["Compost", "Recycle", "Landfill"]

# Fake prediction function (for now)
def predict(image):
    resized = cv2.resize(image, (224, 224))
    resized = resized / 255.0
    resized = np.expand_dims(resized, axis=0)
    
    # Simulate random predictions
    prediction = np.random.rand(3)
    prediction = prediction / np.sum(prediction)
    return prediction

# Open the webcam (with correct backend for Mac)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Flip the frame (optional, feels natural)
    frame = cv2.flip(frame, 1)

    # Display instructions
    cv2.putText(frame, "Press 'c' to classify object. Press 'q' to quit.", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Trash Classifier', frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    if key == ord('c'):
        # When 'c' is pressed, predict
        predictions = predict(frame)
        class_idx = np.argmax(predictions)
        label = labels[class_idx]
        confidence = predictions[class_idx]

        # Draw a rectangle around the object
        height, width, _ = frame.shape
        start_point = (int(width * 0.2), int(height * 0.2))
        end_point = (int(width * 0.8), int(height * 0.8))
        color = (0, 255, 0)  # Green rectangle
        thickness = 4
        cv2.rectangle(frame, start_point, end_point, color, thickness)

        # Put the label and confidence
        label_text = f"{label} ({confidence * 100:.2f}%)"
        cv2.putText(frame, label_text, (start_point[0], start_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Show classified frame for 3 seconds
        cv2.imshow('Trash Classifier', frame)
        cv2.waitKey(3000)  # Wait 3 seconds

cap.release()
cv2.destroyAllWindows()
