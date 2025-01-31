import numpy as np

def myFresnel(hologram, reconstruction_distance, wavelength, pixel_size):
    # Written by Logan Williams, 4/27/2012
    # This function reconstructs a hologram using a Fresnel transform
    # and returns the reconstructed hologram matrix

    H = hologram
    d = reconstruction_distance
    w = wavelength
    dx = pixel_size

    # Use double precision to allow for complex numbers
    H = np.array(H, dtype=np.float64)
    n = H.shape[0]  # size of hologram matrix nxn
    dy = dx
    Hr = np.zeros((n, n), dtype=np.complex128)  # reconstructed H (pre-allocate memory)
    E = np.zeros((n, n), dtype=np.complex128)  # exponential term (pre-allocate memory)

    k = np.arange(-n/2, n/2)  # array same dimensions as hologram
    l = np.arange(-n/2, n/2)
    [XX,YY] = np.meshgrid(k,l)
    E = np.exp((-1j * np.pi / (w * d)) * ((XX * dx) ** 2 + (YY * dy) ** 2))

    # Reconstruction becomes complex valued
    Hr = np.fft.fftshift(np.fft.fft2(H * E))

    return Hr  # end function

