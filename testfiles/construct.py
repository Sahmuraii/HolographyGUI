from PIL import Image
import numpy as np
import os

def crop_to_square(image):
    width, height = image.size

    # Determine the size of the square crop
    crop_size = min(width, height)

    # Calculate the coordinates for the crop
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = (width + crop_size) // 2
    bottom = (height + crop_size) // 2

    # Crop the image
    return image.crop((left, top, right, bottom))

def generate_contrast_image(ref_image, raw_image):
    ref_image = crop_to_square(ref_image)
    raw_image = crop_to_square(raw_image)

    refDat = np.array(ref_image)  # Convert reference image to a numpy array for data manipulation.
    rawDat = np.array(raw_image)  # Convert raw image to a numpy array for data manipulation.

    # Subtract refDat from rawDat (element-wise) in order to get the contrast image data
    contrast_data = (refDat - rawDat) 

    # Clip the values to ensure they are within [0, 255]
    contrast_data = np.clip(contrast_data, 0, 255)

    # Convert the result back to a PIL image for saving and display
    contrast_image = Image.fromarray(contrast_data.astype(np.uint8))

    # Save the contrast image
    return contrast_image

if __name__ == "__main__":  
    raw_image = Image.open("raw_image.bmp")
    nRBC_folder = "nRBC/nRBC"

    for filename in os.listdir(nRBC_folder):
        if filename.endswith(".bmp"):
            reference_image_path = os.path.join(nRBC_folder, filename)
            ref_image = Image.open(reference_image_path)
            contrast_image = generate_contrast_image(raw_image, ref_image)
            contrast_image.save(os.path.join(nRBC_folder, f"contrast_{filename}"))