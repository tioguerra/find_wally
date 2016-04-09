#
# Where is Wally?
# ===============
# (a practical tutorial on crosscorrelation)
#
# by Rodrigo Guerra
# ------------------------------------------
#

# Import the needed libraries
from PIL import Image, ImageDraw
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

# Adjust image resolution
mpl.rc("savefig", dpi=160)

# Read the images
wally_img = Image.open("wally.jpg")
the_crowd_img = Image.open("the_crowd.jpg")

# Transform the images into NumPy arrays (matrices)
wally = np.array(wally_img).astype(float)
the_crowd = np.array(the_crowd_img).astype(float)

# Normalize inputs
wally = (wally / 255.0) * 2 - 1
the_crowd = (the_crowd / 255.0) * 2 - 1

# Get the dimensions of the images
w, h, channels = wally.shape
W, H, _ = the_crowd.shape
n = float(w * h * channels)

# Calculates the crosscorrelation
cross_corr = np.zeros((W - w, H - h))
for x in range(W - w):
    for y in range(H - h):
        m = np.multiply(the_crowd[x:x+w,y:y+h,:], wally)
        s = np.sum(m)
        cross_corr[x,y] = float(s) / n

# Find the maximum crosscorrelation coordinates
wally_pos = np.unravel_index(np.argmax(cross_corr),cross_corr.shape)

# Normalize results
smallest = np.min(cross_corr)
largest = np.max(cross_corr)
cross_corr = (cross_corr - smallest) / (largest - smallest)

# Displays the crosscorrelation image
plt.imshow(cross_corr)
plt.show()

# Shows the resulting matched position
draw = ImageDraw.Draw(the_crowd_img)
x,y = wally_pos
x = x + w/2 # Adjusting coordinates to point
y = y + h/2 # to the center instead of corner
r = int(np.sqrt(w**2 + h**2))
for i in range(15):
    r = r + 0.5
    draw.ellipse((y-r, x-r, y+r, x+r))
plt.imshow(the_crowd_img)
plt.show()

