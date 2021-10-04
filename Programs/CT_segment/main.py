import time
import tkinter as Tk
import PIL.Image as Image
import cv2
import matplotlib
import numpy as np
from GUI import GUI_Panel
from matplotlib import cm
import img_processing
import matplotlib.pylab as plt


FOLDER_PATH = "E:/Egyetem/AI/_Orvosi képfeldolgozás/Datasets/positive_lung_CT/tudodaganat/"
featured_cmaps = ["bone","hot","twilight","PuBuGn","inferno","seismic","hsv","twilight_shifted","spring","Accent","bwr","afmhot"]
print("App starting...\n")
CT_kepsorozat = img_processing.get_pixels_hu(img_processing.load_CT(FOLDER_PATH))
print("CT_kepsorozat betöltve.\n")

plt.figure(1)
plt.imshow(CT_kepsorozat[100],cmap="bone")
plt.show()

img_n = cv2.normalize(src=CT_kepsorozat[169], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
plt.figure(2)
plt.imshow(Image.fromarray(img_n))
plt.show()

internal = img_processing.get_internal_structures(CT_kepsorozat)
img_int = cv2.normalize(src=internal[169], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

plt.figure(3)
plt.imshow(Image.fromarray(img_int))
plt.show()

"""

my_gui = GUI_Panel(CT_kepsorozat)


my_gui.log("kívül is")


plt.imshow(internal[100])
plt.show()



#my_gui.image_change("rs.jpg")

my_gui.root.mainloop()


print("Preprocessing...\n")
my_gui = GUI_Panel(CT_kepsorozat)
print("Preprocessing befejezve, GUI indul.\n")
my_gui.root.mainloop()
"""