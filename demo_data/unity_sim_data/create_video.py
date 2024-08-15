import cv2
import os

# Path to the directory containing the images
image_folder = 'output_frames'

# Path to save the output video
video_path = 'result_video_improved_results2.mp4'

# Get all image files in the directory
images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
images.sort()  # Ensure the images are in the correct order

# Get the dimensions of the first image to set the video size
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
size = (width, height)

# Create a VideoWriter object
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    out.write(frame)  # Write the frame to the video

# Release the VideoWriter object
out.release()

print(f"Video saved to {video_path}")
