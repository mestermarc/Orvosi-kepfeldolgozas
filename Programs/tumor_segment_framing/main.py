import copy
from datetime import time

import pydicom as dicom
import matplotlib.pylab as plt
from PIL import Image

from datetime import datetime
import numpy as np
import preprocessing as pre
import tumor
from GUI import GUI_Panel


# small dataset:
# FOLDER_PATH = "E:/Egyetem/AI/_Orvosi képfeldolgozás/Datasets/pos_lung_CT_10/tudodaganat/"

# bercimellkas:
from ellipsoid_fit import ellipsoid_acc
dateTimeObj = datetime.now()
print("Time at start:", dateTimeObj)

FOLDER_PATH = "E:/Egyetem/AI/_Orvosi képfeldolgozás/Datasets/Berci_mellkas/"
featured_cmaps = ["bone", "hot", "twilight", "PuBuGn", "inferno", "seismic", "hsv", "twilight_shifted", "spring",
                  "Accent", "bwr", "afmhot"]

# 0 - prepare and load DICOM images
# 1 - load small dataset for testing
# 2 - load tumors- full dataset
# 3 - GUI usage
load_DICOM = 1

if load_DICOM == 0:
    CT_dicom = pre.load_CT(FOLDER_PATH)
    CT_kepsorozat = pre.get_pixels_hu(CT_dicom)
    print("CT_kepsorozat ndarray is ready.")

    small_internal = pre.get_internal_structures(CT_kepsorozat)
    cropped_dataset = pre.crop_LUNG_dataset(small_internal, 500)
    cropped_CT = pre.crop_rgb_LUNG_dataset(CT_kepsorozat, 500, small_internal)

    internal_dataset = pre.cutoffborder(cropped_dataset)
    #print("internal_dataset is ready.\n preprocessing finished.")
    np.save('bercidataset.npy', internal_dataset)
    #print("bercidataset.npy saved.")

    np.save('cropped_CT.npy', cropped_CT)
    #print("cropped_CT.npy saved.")

    np.save('bercidataset_small.npy', internal_dataset[110:120])
    #print("bercidataset_small.npy saved.")

    np.save('cropped_CT_small.npy', cropped_CT[110:120])
    #print("cropped_CT_small.npy saved.")

elif load_DICOM == 1:
    internal_dataset = np.load('bercidataset_small.npy')
    print("Opened presaved ndarray (bercidataset_small)")

    cropped_CT = np.load('cropped_CT_small.npy')
    print("Opened presaved ndarray (cropped_CT_small)")

elif load_DICOM == 2:
    internal_dataset = np.load('bercidataset.npy')
    print("Opened presaved ndarray (bercidataset)")

    cropped_CT = np.load('cropped_CT.npy')
    #print("Opened presaved ndarray (cropped_CT)")
else:
    CT_kepsorozat = pre.get_pixels_hu(pre.load_CT(FOLDER_PATH))
    print("CT_kepsorozat betöltve.\n")
    print("Preprocessing...\n")
    my_gui = GUI_Panel(CT_kepsorozat)

    print("Preprocessing befejezve, GUI indul.\n")
    my_gui.log("Lung segmentation")
    my_gui.root.mainloop()

print("dataset size is: {}".format(len(internal_dataset)))
dateTimeObj = datetime.now()
print("Time after reading:", dateTimeObj)
# mode setting:
# 0: full dataset plot tumors in every single image
# 1: full dataset summed in one
# 2: full dataset to 3 sums
# 3: plot 3D
# 4: plotly

MODE = 2

tumors = []

if MODE == 0:

    #   tumor.plot_all_sus(tumors)
    #   for pic in cropped_CT:
    #   plt.imshow(pic, cmap="bone")

    for i in range(0, len(internal_dataset)):
        # frames all the "circlish" shapes in every slide
        pre.segment_frame_plot(tumors, internal_dataset[i], cropped_CT[i], 50, 800, 15, False)

    plot_all = False
    tumor.plot_all_sus(tumors, plot_all)

elif MODE == 1:

    PLOT_ENABLED = False
    for i in range(3, 7):
        print("{}. image:".format(i))
        print(np.shape(internal_dataset[2]))
        pre.segment_frame_plot(tumors, internal_dataset[i], cropped_CT[i], 1, 1000, 5, PLOT_ENABLED, i)

    tumor.plot_all_sus(tumors, False)

elif MODE == 2:
    PLOT_ENABLED = False
    for i in range(0, len(internal_dataset) - 1):
        #print("{}. image:".format(i))
        #print(np.shape(internal_dataset[2]))
        pre.segment_frame_plot(tumors, internal_dataset[i], cropped_CT[i], 3, 1000, 5, PLOT_ENABLED, i)

    # tumor.plot_all_sus(tumors, False)

    # tumor.plot_sus(tumors)

elif MODE == 3:
    print("plot3D")
    # inter = internal_dataset[15:60]
    # pre.plot_3d(internal_dataset[::-1])
    pre.plot_3d(internal_dataset[::-1])

elif MODE == 4:
    pre.slicer(internal_dataset)

print("Found {} forms:".format(len(tumors)))
dateTimeObj = datetime.now()
print("Time after region detection:", dateTimeObj)
tumors = [tumor for tumor in tumors if tumor.getLenght() >= 3]
# tumors = [tumor for tumor in tumors if tumor.get_proba()]
print("Found {} big enough suspicious forms:".format(len(tumors)))

METHOD = "CIRLCE SHAPED MORE"

for tm in tumors:
    tm.calculate_proba(METHOD)


dateTimeObj = datetime.now()
print("Time after probability calculating:", dateTimeObj)
tumors = [tumor for tumor in tumors if tumor.get_proba() > 0.8]

print("Found {} REALLY suspicious -above 80% forms:".format(len(tumors)))
tumor.ellipsoid_fitting(tumors)


dateTimeObj = datetime.now()
print("Time at end:", dateTimeObj)