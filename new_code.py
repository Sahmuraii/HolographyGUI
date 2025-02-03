import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import trim_mean
from scipy.fft import fft
import matplotlib.pyplot as plt
from PIL import Image

cm = 1.0*10**-2
um = 1.0*10**-6
# Constants for reconstruction
z = 25 # Propagation distance
wavelength = .52*um # Wavelength (adjust this value as needed)
pix = 3.45*um  # Pixel size (adjust this value as needed)

# Load the BMP files
import11 = Image.open("1.bmp")
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

# ------------------- Reconstruction -------------------
holoDat = contrast

# Use full resolution for reconstruction
dimX = holoDat.shape[0]
dimY = holoDat.shape[1]

# Define the grid points
x = np.linspace(-dimX / 2, dimX / 2, dimX)
y = np.linspace(-dimY / 2, dimY / 2, dimY)

# Create the interpolator for holoDat1
holo1 = RegularGridInterpolator((y, x), contrast)

# Generate new grid points for interpolation (full resolution)
xx, yy = np.meshgrid(x, y, indexing="ij")
holoDat1 = holo1((yy, xx))

# Calculate holoDat2 (background-subtracted hologram)
aveDat = trim_mean(holoDat1.flatten(), 0.33)  # Trimmed mean of the hologram data
holoDat2 = holoDat1 - aveDat

# Create the grid points for reconstruction
i = np.arange(1, dimX + 1)
j = np.arange(1, dimY + 1)
ii, jj = np.meshgrid(i, j, indexing='ij')

# Compute the exponent term
exponent = (1j * np.pi / wavelength * z) * ((ii - 1)**2 * pix**2 + (jj - 1)**2 * pix**2)

# Evaluate holoDat2 at the grid points (full resolution)
points = np.column_stack((ii.ravel(), jj.ravel()))  # Combine into (i, j) pairs
hologram_values = holoDat2.reshape(-1)  # Use holoDat2 directly (no interpolation needed)

# Compute recon1
recon1 = hologram_values * np.exp(exponent.ravel())

# Reshape recon1 back to 2D
recon1 = recon1.reshape(ii.shape)

# Apply "Chop" by setting small values to zero
recon1 = np.where(np.abs(recon1) < 10**-10, 0, recon1)

# Apply FFT (full resolution)
recon = np.fft.fft2(recon1)

# Compute the squared magnitude of the reconstructed field
abs_recon_squared = np.abs(recon)**2

# Define the grid points for interpolation
window = dimX / 2  # Use full resolution for the window
x = np.linspace(-window, window, abs_recon_squared.shape[0])  # Grid points along the x-axis
y = np.linspace(-window, window, abs_recon_squared.shape[1])  # Grid points along the y-axis

# Create the interpolator for the reconstructed field
view1 = RegularGridInterpolator((x, y), abs_recon_squared)

# Define the plotting grid
x_plot = np.linspace(-window, window, 2048)  # High-resolution grid for plotting
y_plot = np.linspace(-window, window, 2048)  # High-resolution grid for plotting
xx_plot, yy_plot = np.meshgrid(x_plot, y_plot, indexing='ij')

# Evaluate the interpolated reconstructed field (negative of view1)
z_plot = -view1((xx_plot, yy_plot))

# Plot the reconstructed image
plt.figure(figsize=(6, 6))
plt.imshow(z_plot, cmap='gray', extent=[-window, window, -window, window], origin='lower')
plt.colorbar(label="Intensity")
plt.xlabel("x [pix]")
plt.ylabel("y [pix]")
plt.title("Reconstructed Image (Negative of View)")
plt.show()