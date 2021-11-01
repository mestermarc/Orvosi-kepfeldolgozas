import copy

import pydicom as dicom
import matplotlib.pylab as plt
from PIL import Image

import os
import numpy as np
import preprocessing as pre
import tumor

# small dataset:
# FOLDER_PATH = "E:/Egyetem/AI/_Orvosi képfeldolgozás/Datasets/pos_lung_CT_10/tudodaganat/"

# bercimellkas:
FOLDER_PATH = "E:/Egyetem/AI/_Orvosi képfeldolgozás/Datasets/Berci_mellkas/"
featured_cmaps = ["bone", "hot", "twilight", "PuBuGn", "inferno", "seismic", "hsv", "twilight_shifted", "spring",
                  "Accent", "bwr", "afmhot"]

# load DICOM images:
load_DICOM = False

if (load_DICOM):
    CT_dicom = pre.load_CT(FOLDER_PATH)
    CT_kepsorozat = pre.get_pixels_hu(CT_dicom)
    print("CT_kepsorozat ndarray is ready.")

    small_internal = pre.get_internal_structures(CT_kepsorozat)
    cropped_dataset = pre.crop_LUNG_dataset(small_internal, 500)
    cropped_CT = pre.crop_rgb_LUNG_dataset(CT_kepsorozat, 500, small_internal)

    internal_dataset = pre.cutoffborder(cropped_dataset)
    print("internal_dataset is ready.\n preprocessing finished.")
    np.save('bercidataset.npy', internal_dataset)
    print("bercidataset.npy saved.")

    np.save('cropped_CT.npy', cropped_CT)
    print("cropped_CT.npy saved.")


else:
    internal_dataset = np.load('bercidataset.npy')
    print("Opened presaved ndarray (bercidataset)")

    cropped_CT = np.load('cropped_CT.npy')
    print("Opened presaved ndarray (cropped_CT)")

print("dataset size is: {}".format(len(internal_dataset)))

# mode setting:
# 0: full dataset plot tumors in every single image
# 1: full dataset summed in one
# 2: full dataset to 3 sums
# 3: plot 3D
# 4: plotly

MODE = 2

tumors = []
"""
for pic in internal_dataset[35:40]:
    plt.imshow(pic, cmap="bone")
    plt.show()
"""

if MODE == 0:

    #   tumor.plot_all_sus(tumors)
    #   for pic in cropped_CT:
    #   plt.imshow(pic, cmap="bone")

    for i in range(0, len(internal_dataset)):
        # frames all the "circlish" shapes in every slide
        pre.segment_frame_plot(tumors, internal_dataset[i], cropped_CT[i], 50, 500, 15, False)

    plot_all = False
    tumor.plot_all_sus(tumors, plot_all)

elif MODE == 1:
    sum_pic = pre.sum_pics(internal_dataset)

    pre.segment_frame_plot(tumors, sum_pic, 50, 500, 30)
    plot_all = True
    # tumor.plot_all_sus(tumors, plot_all)
    MASK = True
    # tumor.plot_all(tumors,MASK, croppedOne2 )

elif MODE == 2:
    elsodaganat = internal_dataset[112:117]
    masodikdaganat = internal_dataset[114:118]

    SHOW_ALL = True
    i = 112
    for pic in elsodaganat:
        pre.segment_frame_plot(tumors, pic, cropped_CT[i], 50, 500, 15, True)
        i += 1

    i = 114
    for pic in masodikdaganat:
        pre.segment_frame_plot(tumors, pic, cropped_CT[i], 50, 500, 15, True)
        i += 1

elif MODE == 3:
    print("plot3D")
    inter = internal_dataset[15:60]
    pre.plot_3d(inter)

elif MODE == 4:
    pre.plotly_img(internal_dataset)

print("Found {} suspicious forms:".format(len(tumors)))
