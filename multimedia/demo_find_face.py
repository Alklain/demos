# Importing OpenCV package
import cv2
import os

def extract_frames_opencv(video_path, output_folder):
    try:
        # Checking if the video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file '{video_path}' not found.")
        # Creating output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Capturing video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Error opening video file.")
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Saving each frame as an image file
            frame_filename = os.path.join(
                output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()
        print(
            f"Extraction complete. {frame_count} frames extracted to '{output_folder}'.")

    except FileNotFoundError as e:
        print(f"File Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")



video_path = 'C:/wrk/multimedia_demo/IMG_8181.MOV'

extract_frames_opencv(video_path, 'frame_to_wrk')
# Reading the image
fname = 'frame_to_wrk/frame_0065.jpg'

img = cv2.imread(fname)

# Converting image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Loading the required haar-cascade xml classifier file
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#haar_cascade.
# Applying the face detection method on the grayscale image
faces_rect = haar_cascade.detectMultiScale(gray_img, 1.01, 1)

# Iterating through rectangles of detected faces
for (x, y, w, h) in faces_rect:
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow(fname, img)

# Calling the custom function
# Passing sample.mp4 as video and output_folder frames
#extract_frames_opencv('photo/IMG_9155.MOV', 'frames')



cv2.waitKey(0)
