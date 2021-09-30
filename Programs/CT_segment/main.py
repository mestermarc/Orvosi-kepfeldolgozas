import time
import tkinter as Tk
import PIL.Image as Image
from matplotlib import cm
import numpy as np
from GUI import GUI_Panel
import img_processing
import matplotlib.pylab as plt

FOLDER_PATH = "E:/Egyetem/AI/_Orvosi képfeldolgozás/Datasets/positive_lung_CT/tudodaganat/"
featured_cmaps = ["bone","hot","twilight","PuBuGn","inferno","seismic","hsv","twilight_shifted","spring","Accent","bwr","afmhot"]
print("App starting...\n")
CT_kepsorozat = img_processing.get_pixels_hu(img_processing.load_CT(FOLDER_PATH))
print("CT_kepsorozat betöltve.\n")

#pil_image = Image.fromarray(CT_kepsorozat[100], 'RGB')

plt.imshow(CT_kepsorozat[100])
plt.show()


"""
print("Preprocessing...\n")
my_gui = GUI_Panel(CT_kepsorozat)

print("Preprocessing befejezve, GUI indul.\n")
my_gui.log("kívül is")


plt.imshow(internal[100])
plt.show()



#my_gui.image_change("rs.jpg")

my_gui.root.mainloop()
"""
my_gui = GUI_Panel(CT_kepsorozat)

my_gui.root.mainloop()
