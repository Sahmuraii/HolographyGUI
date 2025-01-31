import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import trim_mean
from scipy.fft import fft
import matplotlib.pyplot as plt
from PIL import Image

cm = 1.0*10**-2
um = 1.0*10**-6
# Constants for reconstruction
z = 6 # Propagation distance
wavelength = .52*um # Wavelength (adjust this value as needed)
pix = 3.45*um  # Pixel size (adjust this value as needed)

# Load the BMP files
import11 = Image.open("6cm.bmp")
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

dimX = holoDat.shape[0] // 2
dimY = holoDat.shape[1] // 2

# Define the grid points
x = np.linspace(-dimX, dimX, holoDat.shape[0])  # Grid points along the x-axis
y = np.linspace(-dimY, dimY, holoDat.shape[1])  # Grid points along the y-axis

# Create the interpolator
holo1 = RegularGridInterpolator((x, y), holoDat)

# Generate new grid points for interpolation
new_x = np.arange(-dimX, dimX + 1)  # New x grid points
new_y = np.arange(-dimY, dimY + 1)  # New y grid points

# Create a meshgrid for the new points
new_xx, new_yy = np.meshgrid(new_x, new_y, indexing='ij')

# Evaluate the interpolator at the new points
#points = np.column_stack((new_xx.ravel(), new_yy.ravel()))  # Combine into (x, y) pairs
#holoDat1 = holo1(points).reshape(new_xx.shape)  # Reshape to match the grid

aveDat = trim_mean(holoDat1.flatten(), 0.33)  # Trimmed mean of the hologram data
holoDat2 = holoDat1 - aveDat


dim = len(holoDat1)  # Get the dimensions of

# Create the grid points
x = np.arange(1, dim + 1)
y = np.arange(1, dim + 1)

# Create the interpolator
hologram1 = RegularGridInterpolator((x, y), holoDat2)
recon1 = np.zeros((dim, dim), dtype=complex)  # Initialize the result array

# Create grids for i and j (1-based indexing)
i = np.arange(1, dim + 1)
j = np.arange(1, dim + 1)
ii, jj = np.meshgrid(i, j, indexing='ij')

# Compute the exponent term
exponent = (1j * np.pi / wavelength * z) * ((ii - 1)**2 * pix**2 + (jj - 1)**2 * pix**2)

# Evaluate hologram1 at the grid points (i, j)
# RegularGridInterpolator expects input as a 2D array of shape (N, 2)
points = np.column_stack((ii.ravel(), jj.ravel()))  # Combine into (i, j) pairs
hologram_values = hologram1(points).reshape(ii.shape)  # Evaluate and reshape

# Compute recon1
recon1 = hologram_values * np.exp(exponent)

# Apply "Chop" by setting small values to zero
recon1 = np.where(np.abs(recon1) < 10^-10, 0, recon1)
recon = np.fft.fft2(recon1)
window = dim / 2

abs_recon_squared = np.abs(recon)**2

# Define the grid points for interpolation
x = np.linspace(-window, window, abs_recon_squared.shape[0])  # Grid points along the x-axis
y = np.linspace(-window, window, abs_recon_squared.shape[1])  # Grid points along the y-axis

# Create the interpolator
view1 = RegularGridInterpolator((x, y), abs_recon_squared)

x_plot = np.linspace(-window, window, 600)  # 300 points along x-axis
y_plot = np.linspace(-window, window, 600)  # 300 points along y-axis
xx, yy = np.meshgrid(x_plot, y_plot, indexing='ij')

# Evaluate view1 at the grid points
z_plot = -view1((xx, yy))  # Negative of view1, as in Mathematica

# Shift the zero-frequency component to the center
z_shifted = np.fft.fftshift(z_plot)

# Normalize the shifted data
z_shifted_normalized = (z_shifted - np.min(z_shifted)) / (np.max(z_shifted) - np.min(z_shifted))

# Create a grayscale plot with the shifted data
plt.figure(figsize=(10, 8))
plt.imshow(z_shifted_normalized, cmap='gray', origin='lower')
plt.xlabel("x[pix]", fontsize=30)
plt.ylabel("y[pix]", fontsize=30)
plt.title("reconstructed image")
plt.axis('on')  # Show axes for reference
plt.savefig('reconstructed_image_fftshifted.png', bbox_inches='tight', pad_inches=0, dpi=300)  # Save as PNG
plt.close()  # Close the plot to free memory

# Create the density plot
plt.figure(figsize=(6, 6))  # Set image size (600x600 in Mathematica corresponds to 10x8 in matplotlib)
plt.imshow(z_plot, extent=[-window, window, -window, window], 
           cmap='gray', origin='lower',)  # Create the density plot

# Add labels and title
plt.xlabel("x[pix]")
plt.ylabel("y[pix]")
plt.title("reconstructed image")

# Customize the plot
plt.colorbar(label="Intensity")  # Add a colorbar
plt.xticks()  # Set x-axis tick font
plt.yticks()  # Set y-axis tick font
plt.grid(False)  # Disable grid (Mathematica's DensityPlot doesn't show a grid by default)

# Show the plot
plt.show()
