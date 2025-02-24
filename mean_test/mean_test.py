import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from PIL import Image
from pyDHM import numericalPropagation
from skimage.filters import threshold_otsu
from skimage.morphology import closing, opening, square
from skimage.measure import label, regionprops

# Constants
cm = 1.0 * 10**-2
um = 1.0 * 10**-6
nm = 1.0 * 10**-9
z = 25  # Propagation distance
wavelength = 532 * nm  # Wavelength
pix = 3.45 * um  # Pixel size

# Load 5 sample images
sample_paths = ["image1.bmp", "image2.bmp", "image3.bmp", "image4.bmp", "image5.bmp"]
sample_images = [Image.open(path) for path in sample_paths]

# Load 5 background images
background_paths = ["background1.bmp", "background2.bmp", "background3.bmp", "background4.bmp", "background5.bmp"]
background_images = [Image.open(path) for path in background_paths]

# Crop all images to 2048 x 2048
desired_size = (2048, 2048)
sample_images = [img.crop((0, 0, desired_size[0], desired_size[1])) for img in sample_images]
background_images = [img.crop((0, 0, desired_size[0], desired_size[1])) for img in background_images]

# Convert all images to grayscale
sample_images = [img.convert("L") for img in sample_images]
background_images = [img.convert("L") for img in background_images]

# Ensure all images have the same dimensions
if any(img.size != desired_size for img in sample_images + background_images):
    raise ValueError("Error: All images must have the same dimensions.")

# Convert images to numpy arrays
sample_data = [np.array(img, dtype=int) for img in sample_images]
background_data = [np.array(img, dtype=int) for img in background_images]

# Calculate the mean of the sample and background images
mean_sample = np.mean(sample_data, axis=0)
mean_background = np.mean(background_data, axis=0)

# Calculate the contrast image
contrast = mean_background - mean_sample

# Get the dimensions of the data
dimY, dimX = contrast.shape  # dimY = 2048, dimX = 2048

# Create grids for interpolation
x = np.linspace(-dimX / 2, dimX / 2, dimX)
y = np.linspace(-dimY / 2, dimY / 2, dimY)

# Interpolate the hologram data
holo1 = RegularGridInterpolator((y, x), contrast)

# Create a grid of interpolated values
xx, yy = np.meshgrid(x, y, indexing="ij")
holoDat1 = holo1((yy, xx))

# Plot the contrast hologram
plt.figure(figsize=(6, 6))
plt.imshow(holoDat1, cmap="gray", extent=[-dimX / 2, dimX / 2, -dimY / 2, dimY / 2])
plt.colorbar(label="Intensity")
plt.xlabel("x [pix]")
plt.ylabel("y [pix]")
plt.title("Contrast Hologram")
plt.show()

# Save the contrast hologram
gray_image = Image.fromarray(holoDat1)
gray_image = gray_image.convert("L")
gray_image.save("contrast_hologram.png")

# ------------------- Reconstruction -------------------

# Convert the hologram data to a complex field
complex_field = holoDat1.astype(np.complex128)

# Perform numerical propagation using the angular spectrum method
reconstructed_field = numericalPropagation.angularSpectrum(complex_field, 60000, .532, 3.5, 3.5)

# Extract the amplitude and phase of the reconstructed field
amplitude = np.abs(reconstructed_field)
phase = np.angle(reconstructed_field)

# Plot the amplitude of the reconstructed field
plt.figure(figsize=(6, 6))
plt.imshow(amplitude, cmap="gray", extent=[-dimX / 2, dimX / 2, -dimY / 2, dimY / 2])
plt.colorbar(label="Intensity")
plt.xlabel("x [pix]")
plt.ylabel("y [pix]")
plt.title("Reconstructed Amplitude (Angular Spectrum)")
plt.show()

# Plot the phase of the reconstructed field
plt.figure(figsize=(6, 6))
plt.imshow(phase, cmap="gray", extent=[-dimX / 2, dimX / 2, -dimY / 2, dimY / 2])
plt.colorbar(label="Phase [rad]")
plt.xlabel("x [pix]")
plt.ylabel("y [pix]")
plt.title("Reconstructed Phase (Angular Spectrum)")
plt.show()

# Save the reconstructed amplitude and phase images
amplitude_image = Image.fromarray((amplitude / np.max(amplitude) * 255).astype(np.uint8))
amplitude_image = amplitude_image.convert("L")
amplitude_image.save("reconstructed_amplitude_angular_spectrum.png")

phase_image = Image.fromarray((phase / np.max(phase) * 255).astype(np.uint8))
phase_image = phase_image.convert("L")
phase_image.save("reconstructed_phase_angular_spectrum.png")

# Create a figure to display the images in a grid
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot the contrast image
axs[0].imshow(holoDat1, cmap="gray", extent=[-dimX / 2, dimX / 2, -dimY / 2, dimY / 2])
axs[0].set_title("Contrast Image")
axs[0].set_xlabel("x [pix]")
axs[0].set_ylabel("y [pix]")

# Plot the reconstructed amplitude
axs[1].imshow(amplitude, cmap="gray", extent=[-dimX / 2, dimX / 2, -dimY / 2, dimY / 2])
axs[1].set_title("Reconstructed Amplitude")
axs[1].set_xlabel("x [pix]")
axs[1].set_ylabel("y [pix]")

# Plot the reconstructed phase
axs[2].imshow(phase, cmap="gray", extent=[-dimX / 2, dimX / 2, -dimY / 2, dimY / 2])
axs[2].set_title("Reconstructed Phase")
axs[2].set_xlabel("x [pix]")
axs[2].set_ylabel("y [pix]")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# Save the figure with the image grid
fig.savefig("image_grid.png")

# ------------------- Particle Analysis -------------------

# Binarize the phase image using Otsu's method
threshold_value = threshold_otsu(phase)
binary_phase = phase > threshold_value

# Apply morphological operations to clean up the binary image
binary_phase = closing(binary_phase, square(3))  # Closing: Fills small holes
binary_phase = opening(binary_phase, square(3))  # Opening: Removes small objects

# Plot the cleaned binary phase image
plt.figure(figsize=(6, 6))
plt.imshow(binary_phase, cmap="gray", extent=[-dimX / 2, dimX / 2, -dimY / 2, dimY / 2])
plt.title("Binary Phase Image After Morphological Operations")
plt.xlabel("x [pix]")
plt.ylabel("y [pix]")
plt.show()

# Save the cleaned binary phase image
binary_phase_image = Image.fromarray((binary_phase * 255).astype(np.uint8))
binary_phase_image = binary_phase_image.convert("L")
binary_phase_image.save("binary_phase_image.png")