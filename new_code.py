import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from PIL import Image
from pyDHM import numericalPropagation
from skimage.measure import label, regionprops, profile_line
from skimage.filters import threshold_otsu
from matplotlib.widgets import Line2D

import matplotlib.pyplot as plt

cm = 1.0*10**-2
um = 1.0*10**-6
nm = 1.0*10**-9
# Constants for reconstruction
z = 25 # Propagation distance
wavelength = 532*nm # Wavelength (adjust this value as needed)
pix = 3.45*um  # Pixel size (adjust this value as needed)

# Load the BMP files
import11 = Image.open("6.bmp")
import22 = Image.open("background-elongated.bmp")

# Crop the images to 2048 x 2048 if they are not already
desired_size = (2048, 2048)
import11 = import11.crop((0, 0, desired_size[0], desired_size[1]))
import22 = import22.crop((0, 0, desired_size[0], desired_size[1]))

# Convert images to grayscale if they are not already
import11 = import11.convert("L")
import22 = import22.convert("L")

# Ensure the images are of the same dimensions
if import11.size != import22.size:
    raise ValueError("Error: The images must have the same dimensions.")

# Convert images to numpy arrays
import11_data = np.array(import11, dtype=int)
import22_data = np.array(import22, dtype=int)

# Calculate the contrast between the two images
contrast = import22_data - import11_data

# Get the dimensions of the data
dimY, dimX = contrast.shape # dimY = 2448, dimX = 2048

# Create grids for interpolation
x = np.linspace(-dimX / 2, dimX / 2, dimX)
y = np.linspace(-dimY / 2, dimY / 2, dimY)

# Interpolate the hologram data
holo1 = RegularGridInterpolator((y, x), contrast)

# Create a grid of interpolated values
xx, yy = np.meshgrid(x, y, indexing="ij")
holoDat1 = holo1((yy, xx))

# Plot the hologram data
plt.figure(figsize=(6, 6))
plt.imshow(holoDat1, cmap="gray", extent=[-dimX / 2, dimX / 2, -dimY / 2, dimY / 2])
plt.colorbar(label="Intensity")
plt.xlabel("x [pix]")
plt.ylabel("y [pix]")
plt.title("Contrast Hologram")
plt.show()

# Save the gray mapped image
gray_image = Image.fromarray(holoDat1)
gray_image = gray_image.convert("L")
gray_image.save("holoDat1.png")


# ------------------- Reconstruction -------------------

# Convert the hologram data to a complex field
# Assuming the contrast hologram is the real part, and the imaginary part is zero
complex_field = holoDat1.astype(np.complex128)

# Perform numerical propagation using the angular spectrum method
reconstructed_field = numericalPropagation.angularSpectrum(complex_field, 30000, .532, 3.5, 3.5)

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