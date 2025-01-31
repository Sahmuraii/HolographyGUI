from PIL import Image, ImageChops
import numpy as np
from FresnelPropagator import FresnelPropagator
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import trim_mean

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

    # Subtract refDat from rawDat (element-wise) in order to get the contrast image data
    image = ImageChops.subtract(raw_image, ref_image)
    image.show()
    contrast_data = np.array(ImageChops.subtract(raw_image, ref_image))

    dimX, dimY = contrast_data.shape[0] // 2, contrast_data.shape[1] // 2

   
    x = np.linspace(-dimX, dimX, contrast_data.shape[0])
    y = np.linspace(-dimY, dimY, contrast_data.shape[1])

    interpolator = RegularGridInterpolator((x, y), contrast_data)
    x_new = np.linspace(-dimX, dimX, 300)  # 300 points for high resolution
    y_new = np.linspace(-dimY, dimY, 300)
    xx, yy = np.meshgrid(x_new, y_new)
    holoDat1 = np.array(interpolator((xx, yy)))

    
    plt.figure(figsize=(8, 8))  # ImageSize: 600px (scaled with DPI)
    plt.imshow(
        holoDat1, 
        extent=(-dimX, dimX, -dimX, dimX),  # Set axis limits
        origin="lower",  # Match density plot origin
        cmap="gray",  # ColorFunction: GrayLevel 
    )
    plt.colorbar(label="Contrast Hologram")  # Add color bar

    # Add frame and labels
    plt.xlabel("x [pix]", fontsize=16)
    plt.ylabel("y [pix]", fontsize=16)
    plt.title("Contrast Hologram", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Display plot
    plt.show()


    # Clip the values to ensure they are within [0, 255]
    #contrast_data = np.clip(contrast_data, 0, 255)

    # Convert the result back to a PIL image for saving and display
    #contrast_image = Image.fromarray(contrast_data.astype(np.uint8))

    # Save the contrast image
    return image

def generate_red_hologram(contrast, reconstruction_distance, wavelength, pixel_size):
    contrast_red = np.array(contrast.split()[0])
    viewRed = -np.abs(FresnelPropagator(contrast_red, pixel_size, wavelength, reconstruction_distance))
    view_red_image = Image.fromarray(viewRed.astype(np.uint8))
    #plt.imshow(view_red_image, cmap='Reds')
    return view_red_image