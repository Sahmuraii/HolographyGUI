import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from hologram import generate_contrast_image, generate_red_hologram

class HologramGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hologram GUI")
        self.reconstruction_distance = 1000
        self.wavelength = 0.0000006328
        self.pixel_size = 3.45

        self.raw_image = None
        self.reference_image = None
        self.contrast_image = None

        # Create a frame to hold the buttons
        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.BOTTOM, anchor=tk.N)

        # Create and pack the buttons inside the frame
        self.upload_raw_button = tk.Button(button_frame, text="Upload Raw Image", command=self.upload_raw_image)
        self.upload_raw_button.pack(side=tk.LEFT, padx=5)

        self.upload_reference_button = tk.Button(button_frame, text="Upload Reference Image", command=self.upload_reference_image)
        self.upload_reference_button.pack(side=tk.LEFT, padx=5)

        self.generate_contrast_button = tk.Button(button_frame, text="Generate Contrast Image", command=self.generate_contrast_image)
        self.generate_contrast_button.pack(side=tk.LEFT, padx=5)

        self.open_red_hologram_window = tk.Button(button_frame, text="Open Red Hologram Window", command=self.open_red_hologram_window)
        self.open_red_hologram_window.pack(side=tk.LEFT, padx=5)


        self.raw_image_label = tk.Label(root)
        self.raw_image_label.pack()

        self.reference_image_label = tk.Label(root)
        self.reference_image_label.pack()

        self.contrast_image_label = tk.Label(root)
        self.contrast_image_label.pack()

    def upload_raw_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.raw_image = Image.open(file_path)
            tk.messagebox.showinfo("Success", "Image was successfully uploaded")
        else:
            tk.messagebox.showerror("Error", "No image selected")

    def upload_reference_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.reference_image = Image.open(file_path)
            tk.messagebox.showinfo("Success", "Image was successfully uploaded")
        else:
            tk.messagebox.showerror("Error", "No image selected")

    def display_image(self, image, label):
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

    def generate_contrast_image(self):
        if self.raw_image and self.reference_image:
            contrast_image = generate_contrast_image(self.raw_image, self.reference_image)
            self.contrast_image = contrast_image
            display_contrast = contrast_image.resize((500,500))
            self.display_image(display_contrast, self.contrast_image_label)

    def open_red_hologram_window(self):
        new_window = tk.Toplevel(self.root)
        new_window.title("Red Hologram")
        
        # Add a label to display the red hologram
        self.red_hologram_label = tk.Label(new_window)
        self.red_hologram_label.pack()
        
        # Add a button to generate the red hologram
        generate_red_hologram_button = tk.Button(new_window, text="Generate Red Hologram", command=lambda: self.generate_red_hologram(new_window))
        generate_red_hologram_button.pack(pady=10)

    def generate_red_hologram(self, window):
        if self.contrast_image:
            red_hologram = generate_red_hologram(self.contrast_image, self.reconstruction_distance, self.wavelength, self.pixel_size)
            display_red = red_hologram.resize((500,500))
            photo = ImageTk.PhotoImage(display_red)
        
            # Update the label in the new window with the image
            self.red_hologram_label.config(image=photo)
            self.red_hologram_label.image = photo


if __name__ == "__main__":
    root = tk.Tk()
    app = HologramGUI(root)
    root.mainloop()
