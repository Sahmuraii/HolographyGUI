from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
from testfiles.new_code import z_plot, window, pix

pixel_size = pix

# Function to draw a rectangle and get its coordinates
def onselect(eclick, erelease):
    global roi
    roi = (int(eclick.ydata), int(eclick.xdata), int(erelease.ydata), int(erelease.xdata))
    print(f"Selected ROI: {roi}")

# Display the image and allow manual selection
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(z_plot, cmap='gray', extent=[-window, window, -window, window], origin='lower')
ax.set_title("Select the particle region")

# Create a rectangle selector
rs = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)

plt.show()

# After selecting the ROI, use it to measure the particle
if 'roi' in globals():
    minr, minc, maxr, maxc = roi
    length_pixels = maxr - minr
    width_pixels = maxc - minc
    length_um = length_pixels * 3.45 * 10**-6
    width_um = width_pixels * 3.45 *  10**-6
    print(f"Particle length (pixels): {length_pixels}")
    print(f"Particle width (pixels): {width_pixels}")
    print(f"Particle length (µm): {length_um}")
    print(f"Particle width (µm): {width_um}")