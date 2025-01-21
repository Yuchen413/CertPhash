import os
import cv2
import pandas as pd
import pdqhash

# Define the root directory containing images
root = '../train_verify/data/mnist/training'

# Prepare a list to collect data
data = []

# Iterate over all files in the directory
for subdir, dirs, files in os.walk(root):
    for file in files:
        # Construct the full file path
        file_path = os.path.join(subdir, file)

        # Load the image
        image = cv2.imread(file_path)
        if image is None:
            continue  # Skip files that are not images

        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Compute the hash and quality
        hash_vector, quality = pdqhash.compute(image)

        # Extract the relative path starting from 'testing/'
        relative_path = os.path.relpath(file_path, '../data')

        # Append the path and hash_vector to the data list
        data.append([relative_path, hash_vector])

# Create a DataFrame from the collected data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file without headers
df.to_csv(os.path.join('../Normal-Training/mnist','mnist_train.csv'), index=False, header=False)

#I want to imurate all images under root, and generate the hash, then save it with name mnist_test.csv, with two collomns. The first one is the path of the image, start with 'testing/', the second is the hash vector

# # Get all the rotations and flips in one pass.
# # hash_vectors is a list of vectors in the following order
# # - Original
# # - Rotated 90 degrees
# # - Rotated 180 degrees
# # - Rotated 270 degrees
# # - Flipped vertically
# # - Flipped horizontally
# # - Rotated 90 degrees and flipped vertically
# # - Rotated 90 degrees and flipped horizontally
# hash_vectors, quality = pdqhash.compute_dihedral(image)
# # Get the floating point values of the hash.
# hash_vector_float, quality = pdqhash.compute_float(image)