import os
import numpy as np

# Define the transformation matrix
glcam_in_cvcam = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]]).astype(float)

# Create output directory if it doesn't exist
input_dir = 'ob_in_cam'
output_dir = 'ob_in_cam_new'
os.makedirs(output_dir, exist_ok=True)

# Loop through each .txt file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        # Load the transformation matrix from the file
        filepath = os.path.join(input_dir, filename)
        cam_in_obs = np.loadtxt(filepath).reshape(4, 4)

        # Apply the transformation
        glcam_in_obs = np.dot(cam_in_obs, glcam_in_cvcam)

        # Save the transformed matrix to the output directory
        output_filepath = os.path.join(output_dir, filename)
        np.savetxt(output_filepath, glcam_in_obs, fmt='%.6f')

print(f"Transformation applied and saved to '{output_dir}'")
