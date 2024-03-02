import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# Define a list of NIfTI file paths
file_paths = ['tr_im.nii', 'tr_mask.nii']  # Add more file paths as needed

for file_path in file_paths:
    try:
        # Load the NIfTI image and get the data
        test_load = nib.load(file_path).get_fdata()

        # Display the shape of the loaded data
        print(f"Shape of the loaded data ({file_path}):", test_load.shape)

        # Calculate the min and max intensity values (Housnsefield units)
        min_intensity = np.min(test_load)
        max_intensity = np.max(test_load)

        print(f"Min intensity (HU): {min_intensity }")

        # Additional code to visualize or process the data if needed
        test = test_load[:, :, 59]
        plt.imshow(test)
        plt.title(file_path)
        plt.show()

    except FileNotFoundError:
        print(f"File not found: {file_path}")

    except Exception as e:
        print(f"An error occurred for {file_path}: {e}")
