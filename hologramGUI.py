import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from pyDHM import numericalPropagation
from skimage.filters import threshold_otsu
from skimage.morphology import closing, opening, square
from skimage.measure import label, regionprops
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector

# Constants
cm = 1.0 * 10**-2
um = 1.0 * 10**-6
nm = 1.0 * 10**-9

class HologramGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hologram Analyzer")
        self.root.geometry("1200x800")

        # Variables
        self.pix_size = tk.DoubleVar(value=3.45)  # Default pixel size in um
        self.wavelength = tk.DoubleVar(value=532)  # Default wavelength in nm
        self.z = tk.DoubleVar(value=5)  # Default propagation distance in cm

        # GUI Elements
        self.create_widgets()

        # Rectangle selector variables
        self.rect_coords = None
        self.rect_selector = None

        # Ensure the program exits cleanly when the window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        # Frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Pixel size input
        tk.Label(control_frame, text="Pixel Size (um):").grid(row=0, column=0, sticky="w")
        tk.Entry(control_frame, textvariable=self.pix_size).grid(row=0, column=1)

        # Wavelength input
        tk.Label(control_frame, text="Wavelength (nm):").grid(row=1, column=0, sticky="w")
        tk.Entry(control_frame, textvariable=self.wavelength).grid(row=1, column=1)

        # Propagation distance input
        tk.Label(control_frame, text="Propagation Distance (cm):").grid(row=2, column=0, sticky="w")
        tk.Entry(control_frame, textvariable=self.z).grid(row=2, column=1)

        # Buttons
        tk.Button(control_frame, text="Load Raw Image", command=self.load_raw_image).grid(row=3, column=0, columnspan=2, pady=5)
        tk.Button(control_frame, text="Load Background Image", command=self.load_background_image).grid(row=4, column=0, columnspan=2, pady=5)
        tk.Button(control_frame, text="Process Images", command=self.process_images).grid(row=5, column=0, columnspan=2, pady=5)
        tk.Button(control_frame, text="Measure Particles", command=self.activate_rectangle_selector).grid(row=6, column=0, columnspan=2, pady=5)

        # Frame for displaying images
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Placeholder for matplotlib figure (only binarized image)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_raw_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])
        if file_path:
            self.raw_image = Image.open(file_path)
            self.raw_image = self.raw_image.crop((0, 0, 2048, 2048)).convert("L")
            messagebox.showinfo("Info", "Raw image loaded successfully.")

    def load_background_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])
        if file_path:
            self.background_image = Image.open(file_path)
            self.background_image = self.background_image.crop((0, 0, 2048, 2048)).convert("L")
            messagebox.showinfo("Info", "Background image loaded successfully.")

    def process_images(self):
        if not hasattr(self, 'raw_image') or not hasattr(self, 'background_image'):
            messagebox.showerror("Error", "Please load both raw and background images.")
            return

        # Convert images to numpy arrays
        raw_data = np.array(self.raw_image, dtype=int)
        background_data = np.array(self.background_image, dtype=int)

        # Calculate the contrast
        contrast = background_data - raw_data

        # Interpolate the hologram data
        dimY, dimX = contrast.shape
        x = np.linspace(-dimX / 2, dimX / 2, dimX)
        y = np.linspace(-dimY / 2, dimY / 2, dimY)
        holo1 = RegularGridInterpolator((y, x), contrast)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        holoDat1 = holo1((yy, xx))

        # Reconstruct the field
        complex_field = holoDat1.astype(np.complex128)
        reconstructed_field = numericalPropagation.angularSpectrum(
            complex_field, self.z.get() * cm, self.wavelength.get() * nm, self.pix_size.get() * um, self.pix_size.get() * um
        )
        self.phase = np.angle(reconstructed_field)

        # Check if the phase image is valid
        if np.all(np.isnan(self.phase)) or self.phase.size == 0:
            messagebox.showerror("Error", "Phase image is invalid or empty. Check the input images and parameters.")
            return

        # Binarize the phase image
        try:
            threshold_value = threshold_otsu(self.phase)
            self.binary_phase = self.phase > threshold_value
        except ValueError as e:
            messagebox.showerror("Error", f"Failed to binarize the phase image: {str(e)}")
            return

        # Apply morphological operations
        self.binary_phase = closing(self.binary_phase, square(3))
        self.binary_phase = opening(self.binary_phase, square(3))

        # Update plot with the binarized image
        self.ax.clear()
        self.ax.imshow(self.binary_phase, cmap="gray", extent=[-dimX / 2, dimX / 2, -dimY / 2, dimY / 2])
        self.ax.set_title("Binarized Phase Image")
        self.ax.set_xlabel("x [pix]")
        self.ax.set_ylabel("y [pix]")
        self.canvas.draw()

    def activate_rectangle_selector(self):
        if not hasattr(self, 'binary_phase'):
            messagebox.showerror("Error", "Please process the images first.")
            return

        # Clear any existing rectangle selector
        if self.rect_selector:
            self.rect_selector.set_active(False)

        # Activate rectangle selector on the binarized phase image
        self.rect_coords = None
        self.rect_selector = RectangleSelector(
            self.ax, self.on_rectangle_select, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True
        )
        messagebox.showinfo("Info", "Draw a rectangle on the binarized phase image to measure particles.")

    def on_rectangle_select(self, eclick, erelease):
        # Get rectangle coordinates
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.rect_coords = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

        # Measure particles within the rectangle
        self.measure_particles_in_rectangle()

    def measure_particles_in_rectangle(self):
        if self.rect_coords is None:
            return

        # Extract rectangle coordinates
        x1, y1, x2, y2 = self.rect_coords

        # Crop the binary phase image to the rectangle
        cropped_binary_phase = self.binary_phase[y1:y2, x1:x2]

        # Label particles and measure properties
        labeled_image = label(cropped_binary_phase)
        regions = regionprops(labeled_image)

        # Display particle measurements
        result_text = f"Particle Measurements in Rectangle ({x1}, {y1}) to ({x2}, {y2}):\n"
        for region in regions:
            # Adjust centroid coordinates to the original image
            centroid_y, centroid_x = region.centroid
            centroid_x += x1
            centroid_y += y1
            result_text += f"Particle {region.label}: Area = {region.area} pixels, Centroid = ({centroid_x:.2f}, {centroid_y:.2f})\n"

        messagebox.showinfo("Particle Measurements", result_text)

    def on_close(self):
        # Close matplotlib figures and exit the program
        plt.close('all')
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HologramGUI(root)
    root.mainloop()