# Hologram Analyzer

The **Hologram Analyzer** is a Python-based GUI application designed for analyzing hologram images. It allows you to load raw and background images, process them to extract phase information, binarize the phase image, and measure particle sizes and distances. The program also includes a **Mean Mode** for processing multiple images and a **Scale Setting** feature for converting pixel measurements into real-world units.

---

## Features

1. **Load Raw and Background Images**:
   - Load single raw and background images for processing.
   - Supports BMP file format.

2. **Process Images**:
   - Computes the contrast between raw and background images.
   - Reconstructs the phase image using numerical propagation.
   - Binarizes the phase image using Otsu's thresholding and applies morphological operations.

3. **Mean Mode**:
   - Load 5 raw and 5 background images.
   - Computes the mean of the images for improved noise reduction.

4. **Draw Line and Measure**:
   - Draw lines on the binarized phase image to measure distances.
   - Automatically converts pixel measurements into real-world units using a predefined scale.

5. **Set Scale**:
   - Draw a line on the image and input a known distance to calculate the scale (µm/pixel).
   - Use the scale to measure distances in real-world units.

6. **Interactive GUI**:
   - Built with `tkinter` and `matplotlib` for an intuitive user interface.

---

## Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `PIL` (Pillow)
  - `skimage`
  - `pyDHM`
  - `tkinter`

Install the required libraries using `pip`:

```bash
pip install numpy scipy matplotlib pillow scikit-image pyDHM
```

## How to Use
1. Launch the Program
    - Run the script to launch the Hologram Analyzer GUI. 

2. Load Images
    - Click Load Raw Image to load a single raw hologram image.
    - Click Load Background Image to load a single background image.

3. Process Images
    - Click Process Images to:
    - Compute the contrast between raw and background images.
    - Reconstruct the phase image.
    - Binarize the phase image and display it.

4. Mean Mode (Optional)
    - Click Mean Mode to:
    - Load 5 raw and 5 background images.
    - Compute the mean of the images for noise reduction.
    - Process the mean images as usual.

5. Set Scale
    - Click Set Scale:
    - Draw a line on the binarized phase image.
    - Input the known distance (in µm) corresponding to the line.
    - The program calculates the scale (µm/pixel) and stores it for future measurements.

6. Draw Line and Measure
    - Click Draw Line:
    - Draw a line on the binarized phase image.
    - The program calculates the length of the line in pixels and real-world units (µm) using the predefined scale.

7. Exit the Program
    - Close the window or click the close button to exit the program.