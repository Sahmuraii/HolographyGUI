import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from hologram import generate_contrast_image

class HologramGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hologram GUI")

        self.raw_image = None
        self.reference_image = None

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
        image.thumbnail((200, 200))
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

    def generate_contrast_image(self):
        if self.raw_image and self.reference_image:
            contrast_image = generate_contrast_image(self.raw_image, self.reference_image)
            self.display_image(contrast_image, self.contrast_image_label)
        

if __name__ == "__main__":
    root = tk.Tk()
    app = HologramGUI(root)
    root.mainloop()
