import cv2
import face_recognition
import numpy as np

# Load known image
image_path = "/Users/arinzemomife/Desktop/image-path/kids.jpg"
known_image = face_recognition.load_image_file(image_path)
known_face_encoding = face_recognition.face_encodings(known_image)[0]

known_face_encodings = [known_face_encoding]
known_face_names = ["Your Name"]

# Open the video capture
video_capture = cv2.VideoCapture(1)

process_this_frame = True

while True:
    # Capture frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break
    else:
        print("Frame captured successfully")

    # Resize frame to 15% of original size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.15, fy=0.15)
    rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR (OpenCV) to RGB (face_recognition)

    # Debug: Print the shape of the frame to ensure it's being processed correctly
    print(f"Frame shape: {rgb_small_frame.shape}")

    if process_this_frame:
        # Detect faces in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        print(f"Face locations: {face_locations}")  # Debug: Print detected face locations

        # Only proceed if faces are detected
        if len(face_locations) > 0:
            # Detect landmarks for each face
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)
            print(f"Face landmarks: {face_landmarks_list}")  # Debug: Print landmarks

            # Encode the faces using the landmarks
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            print(f"Face encodings: {face_encodings}")  # Debug: Print the face encodings

        else:
            print("No face detected.")
            face_encodings = []

        # Compare detected faces with known faces and assign names
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

        print(f"Detected faces: {face_names}")  # Debug: Print recognized face names

    process_this_frame = not process_this_frame  # Toggle frame processing

    # Draw boxes around faces and label them
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations to original size
        top *= int(1/0.15)
        right *= int(1/0.15)
        bottom *= int(1/0.15)
        left *= int(1/0.15)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with the name
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the video frame with detected faces
    cv2.imshow('Face Recognition', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the camera and close windows
video_capture.release()
cv2.destroyAllWindows()
