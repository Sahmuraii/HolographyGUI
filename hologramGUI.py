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
        self.known_distance = tk.DoubleVar(value=10)  # Known distance in um
        self.scale = None  # Scale (um/pixel)

        # GUI Elements
        self.create_widgets()

        # Line drawing variables
        self.line_coords = None
        self.line_selector = None
        self.line_start = None
        self.line_end = None

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

        # Known distance for scaling
        tk.Label(control_frame, text="Known Distance (um):").grid(row=3, column=0, sticky="w")
        tk.Entry(control_frame, textvariable=self.known_distance).grid(row=3, column=1)

        # Buttons
        tk.Button(control_frame, text="Load Raw Image", command=self.load_raw_image).grid(row=4, column=0, columnspan=2, pady=5)
        tk.Button(control_frame, text="Load Background Image", command=self.load_background_image).grid(row=5, column=0, columnspan=2, pady=5)
        tk.Button(control_frame, text="Process Images", command=self.process_images).grid(row=6, column=0, columnspan=2, pady=5)
        tk.Button(control_frame, text="Draw Line", command=self.activate_line_selector).grid(row=7, column=0, columnspan=2, pady=5)
        tk.Button(control_frame, text="Set Scale", command=self.activate_scale_selector).grid(row=8, column=0, columnspan=2, pady=5)
        tk.Button(control_frame, text="Mean Mode", command=self.mean_mode).grid(row=9, column=0, columnspan=2, pady=5)

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

    def load_multiple_images(self, title):
        file_paths = filedialog.askopenfilenames(title=title, filetypes=[("BMP files", "*.bmp")])
        if file_paths:
            images = []
            for file_path in file_paths:
                img = Image.open(file_path)
                img = img.crop((0, 0, 2048, 2048)).convert("L")
                images.append(np.array(img, dtype=int))
            return images
        return None

    def compute_mean_image(self, images):
        if images:
            return np.mean(images, axis=0)
        return None

    def mean_mode(self):
        raw_images = self.load_multiple_images("Select 5 Raw Images")
        background_images = self.load_multiple_images("Select 5 Background Images")

        if raw_images and background_images and len(raw_images) == 5 and len(background_images) == 5:
            self.raw_image = self.compute_mean_image(raw_images)
            self.background_image = self.compute_mean_image(background_images)
            messagebox.showinfo("Info", "Mean images computed successfully.")
        else:
            messagebox.showerror("Error", "Please select exactly 5 raw and 5 background images.")

    def process_images(self):
        if not hasattr(self, 'raw_image') or not hasattr(self, 'background_image'):
            messagebox.showerror("Error", "Please load both raw and background images.")
            return

        # Convert images to numpy arrays if they are PIL images
        if isinstance(self.raw_image, Image.Image):
            raw_data = np.array(self.raw_image, dtype=int)
        else:
            raw_data = self.raw_image

        if isinstance(self.background_image, Image.Image):
            background_data = np.array(self.background_image, dtype=int)
        else:
            background_data = self.background_image

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

    def activate_line_selector(self):
        if not hasattr(self, 'binary_phase'):
            messagebox.showerror("Error", "Please process the images first.")
            return

        # Clear any existing line selector
        if self.line_selector:
            self.fig.canvas.mpl_disconnect(self.line_selector)

        # Activate line drawing on the binarized phase image
        self.line_start = None
        self.line_end = None
        self.line_coords = None
        self.line_selector = self.fig.canvas.mpl_connect("button_press_event", self.on_line_draw)
        messagebox.showinfo("Info", "Click to draw a line on the binarized phase image.")

    def activate_scale_selector(self):
        if not hasattr(self, 'binary_phase'):
            messagebox.showerror("Error", "Please process the images first.")
            return

        # Clear any existing line selector
        if self.line_selector:
            self.fig.canvas.mpl_disconnect(self.line_selector)

        # Activate line drawing for scale calculation
        self.line_start = None
        self.line_end = None
        self.line_coords = None
        self.line_selector = self.fig.canvas.mpl_connect("button_press_event", self.on_scale_line_draw)
        messagebox.showinfo("Info", "Draw a line to set the scale.")

    def on_line_draw(self, event):
        if event.inaxes != self.ax:
            return

        if self.line_start is None:
            self.line_start = (event.xdata, event.ydata)
        else:
            self.line_end = (event.xdata, event.ydata)
            self.line_coords = (self.line_start, self.line_end)
            self.draw_line()
            self.calculate_line_length()
            self.fig.canvas.mpl_disconnect(self.line_selector)

    def on_scale_line_draw(self, event):
        if event.inaxes != self.ax:
            return

        if self.line_start is None:
            self.line_start = (event.xdata, event.ydata)
        else:
            self.line_end = (event.xdata, event.ydata)
            self.line_coords = (self.line_start, self.line_end)
            self.draw_line()
            self.calculate_scale()
            self.fig.canvas.mpl_disconnect(self.line_selector)

    def draw_line(self):
        if self.line_coords:
            (x1, y1), (x2, y2) = self.line_coords
            self.ax.plot([x1, x2], [y1, y2], color="red", linewidth=2)
            self.canvas.draw()

    def calculate_line_length(self):
        if self.line_coords and self.scale is not None:
            (x1, y1), (x2, y2) = self.line_coords
            pixel_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            real_length = pixel_length * self.scale
            messagebox.showinfo("Line Length", f"Pixel Length: {pixel_length:.2f} pixels\nReal Length: {real_length:.2f} um")
        else:
            messagebox.showerror("Error", "Please set the scale first.")

    def calculate_scale(self):
        if self.line_coords:
            (x1, y1), (x2, y2) = self.line_coords
            pixel_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            self.scale = self.known_distance.get() / pixel_length
            messagebox.showinfo("Scale Set", f"Scale calculated: {self.scale:.4f} um/pixel")

    def on_close(self):
        # Close matplotlib figures and exit the program
        plt.close('all')
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HologramGUI(root)
    root.mainloop()