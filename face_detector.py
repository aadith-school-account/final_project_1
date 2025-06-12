import cv2

# Load Haar cascade for face detection, built in w/ cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


# Open webcam
camera = cv2.VideoCapture(0)

while True:
    #read current frame from webcam
    (grabbed, frame) = camera.read()
    if not grabbed:
        break

    # Convert to grayscale for face detection, faster
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces, rezies + applies face pattern

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected faces and show size
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Show estimated size

        smiles = smile_cascade.detectMultiScale(
            roi_gray, 
            scaleFactor=1.3, 
            minNeighbors=10, 
            minSize=(20, 20)
     )

        if len(smiles) > 0:
            cv2.putText(frame, "Smile detected!", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No smile detected", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        cv2.putText(frame, f"Size: {w}x{h}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)




    # Display the frame with detected faces
    cv2.imshow("Face Size Detector", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
