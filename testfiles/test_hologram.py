import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import trim_mean

def get_difference():
    image1 = cv2.imread('background.bmp')
    image2 = cv2.imread('raw.bmp')
    difference = cv2.subtract(image1, image2)
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]
    image1[mask != 255] = [0, 0, 255]
    image2[mask != 255] = [0, 0, 255]
    cv2.imwrite('difference.png', image1)

    # Read the saved difference image
    contrast_data = cv2.imread('difference.png', cv2.IMREAD_GRAYSCALE)
    #contrast_data = np.array(image1)

    dimX, dimY = contrast_data.shape[0] // 2, contrast_data.shape[1] // 2
   
    x = np.linspace(-dimX, dimX, contrast_data.shape[0])
    y = np.linspace(-dimY, dimY, contrast_data.shape[1])
    interpolator = RegularGridInterpolator((x, y), contrast_data)

    
    x_new = np.linspace(-dimX, dimX, 1000)  # 300 points for high resolution
    y_new = np.linspace(-dimY, dimY, 1000)
    xx, yy = np.meshgrid(x_new, y_new)
    holo1_data = interpolator((xx, yy))
    holo1_data_normalized = (holo1_data - np.min(holo1_data)) / (np.max(holo1_data) - np.min(holo1_data)) * 255
    holoDat1 = holo1_data_normalized.astype(np.uint8)

    
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

    # ------------------- Reconstruction -------------------


    flattened_data = holoDat1.flatten()
    aveDat = trim_mean(flattened_data, proportiontocut=0.33)
    holoDat2=holoDat1-aveDat


    # Assuming holoDat2 is a 2D numpy array and dim is the length of holoDat2
    dim = len(holoDat2)

    # Define the original grid (Mathematica uses 1-based indexing)
    x = np.arange(1, dim + 1)  # Mathematica indices start at 1
    y = np.arange(1, dim + 1)
    xx, yy = np.meshgrid(x, y)

    # Create the interpolator
    hologram1 = RegularGridInterpolator((x, y), holoDat2)
    hologram1_data = hologram1((xx, yy))
    
    um=1.0*10-6
    z=0.81450
    pix=3.45*um
    λ=0.6328*um
    phase_factor = np.exp((1j * np.pi) / (λ * z) * ((xx - 1)**2 * pix**2 + (yy - 1)**2 * pix**2))

    # Perform the reconstruction calculation
    recon1 = hologram1_data * phase_factor

    # Remove small imaginary components (equivalent to Chop in Mathematica)
    recon1 = np.real_if_close(recon1)

    # Perform the 2D Fourier Transform with the equivalent FourierParameters -> {0, -1}
    recon = np.fft.fft2(recon1)

    # Shift the zero-frequency component to the center (optional, for visualization purposes)
    recon = np.fft.fftshift(recon)

    window = dim // 2

    abs_recon_squared = np.abs(recon)**2

    # Define the interpolation grid
    dim = abs_recon_squared.shape[0]  # Assuming a square array
    window = dim * pix / 2  # Define the window based on your parameters
    x = np.linspace(-window, window, dim)
    y = np.linspace(-window, window, dim)

    # Create the interpolator
    view1_interpolator = RegularGridInterpolator((x, y), abs_recon_squared)

    # Optionally, interpolate onto a new grid for visualization
    x_new = np.linspace(-window, window, 300)  # High-resolution grid
    y_new = np.linspace(-window, window, 300)
    xx, yy = np.meshgrid(x_new, y_new)

    # Evaluate the interpolator on the new grid
    view1 = view1_interpolator((xx, yy))

    plt.figure(figsize=(8, 8))  # ImageSize equivalent to 600px with appropriate DPI scaling
    plt.imshow(
    -view1,  # Negate view1 to match Mathematica's `-view1[x, y]`
    extent=(-window, window, -window, window),  # Set axis limits
    origin="lower",  # Match density plot origin
    cmap="gray",  # ColorFunction equivalent to GrayLevel
    aspect="auto"  # AspectRatio -> Automatic
    )

    # Add a color bar
    plt.colorbar(label="Reconstructed Image")

    # Add frame labels and customize font
    plt.xlabel("x [pix]", fontsize=16)
    plt.ylabel("y [pix]", fontsize=16)
    plt.title("Reconstructed Image", fontsize=20)

    # Customize tick labels
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Display the plot
    plt.show()
    

get_difference()