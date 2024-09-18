# Import the necessary libraries for computer vision and face recognition
import cv2  # OpenCV for handling video and image operations
import face_recognition  # face_recognition library for face detection and recognition
import numpy as np  # NumPy for numerical operations

# Load known image to recognize later
image_path = "/Users/YOUR-COMPUTER-NAME/Desktop/image-path/kids.jpg"  # Path to the known image
known_image = face_recognition.load_image_file(image_path)  # Load the image file
known_face_encoding = face_recognition.face_encodings(known_image)[0]  # Encode the face from the image (get the face feature vector)

# List of known face encodings and names
known_face_encodings = [known_face_encoding]  # A list containing the encoding of the known face
known_face_names = ["Your Name"]  # Name corresponding to the known face

# Open the video capture to use your webcam (camera index 1). If you use the built-in camera, try index 0.
video_capture = cv2.VideoCapture(1)

# Variable to control processing every other frame (to save resources)
process_this_frame = True

# Main loop to process video frames and recognize faces
while True:
    # Capture the current frame from the video
    ret, frame = video_capture.read()
    
    # If the frame was not captured successfully, print an error and exit the loop
    if not ret:
        print("Failed to grab frame")
        break
    else:
        print("Frame captured successfully")  # Debug: Confirmation that the frame was captured

    # Resize the frame to 15% of the original size to speed up face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.15, fy=0.15)

    # Convert the resized frame from BGR color (used by OpenCV) to RGB color (used by face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Debug: Print the dimensions of the resized frame for verification
    print(f"Frame shape: {rgb_small_frame.shape}")

    # Only process every other frame to save computational resources
    if process_this_frame:
        # Detect all face locations in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        print(f"Face locations: {face_locations}")  # Debug: Print face locations

        # Check if any faces are detected in the frame
        if len(face_locations) > 0:
            # Detect facial landmarks (like eyes, nose, mouth) for each face detected
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)
            print(f"Face landmarks: {face_landmarks_list}")  # Debug: Print facial landmarks

            # Encode the detected faces into a feature vector for comparison
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            print(f"Face encodings: {face_encodings}")  # Debug: Print the face encodings (feature vectors)

        else:
            # If no faces are detected, set face_encodings to an empty list
            print("No face detected.")
            face_encodings = []

        # Compare each detected face with the known faces
        face_names = []  # List to store the names of the recognized faces
        for face_encoding in face_encodings:
            # Compare the detected face encodings to the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"  # Default name if no match is found

            # Get the distance between the detected face and the known faces
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            # Find the best match (the smallest distance)
            best_match_index = np.argmin(face_distances)

            # If the best match is a match, assign the corresponding name
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Append the recognized name to the list of face names
            face_names.append(name)

        # Debug: Print the names of recognized faces
        print(f"Detected faces: {face_names}")

    # Toggle to process every other frame
    process_this_frame = not process_this_frame

    # Loop through each face location and corresponding name to draw boxes and labels
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Since we resized the frame earlier, we need to scale the face locations back to the original size
        top *= int(1/0.15)
        right *= int(1/0.15)
        bottom *= int(1/0.15)
        left *= int(1/0.15)

        # Draw a green rectangle around the face in the original frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a filled rectangle at the bottom of the face to display the name label
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        # Write the name of the person on the rectangle
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting frame with recognized faces in a window titled "Face Recognition"
    cv2.imshow('Face Recognition', frame)

    # Exit the loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the webcam resource and close any open OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
