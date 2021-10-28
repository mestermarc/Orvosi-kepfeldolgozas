import copy

import pydicom as dicom
import matplotlib.pylab as plt

import os
import numpy as np
import preprocessing as pre
import tumor
#small dataset:
#FOLDER_PATH = "E:/Egyetem/AI/_Orvosi képfeldolgozás/Datasets/pos_lung_CT_10/tudodaganat/"

#bercimellkas:
FOLDER_PATH = "E:/Egyetem/AI/_Orvosi képfeldolgozás/Datasets/Berci_mellkas/"
featured_cmaps = ["bone", "hot", "twilight", "PuBuGn", "inferno", "seismic", "hsv", "twilight_shifted", "spring",
                  "Accent", "bwr", "afmhot"]

CT_dicom = pre.load_CT(FOLDER_PATH)
CT_kepsorozat = pre.get_pixels_hu(CT_dicom)

plt.imshow(CT_kepsorozat[3])
plt.show()

small_internal = pre.get_internal_structures(CT_kepsorozat)

print("dataset size is: {}".format(len(small_internal)))
plt.imshow(small_internal[3])
plt.show()

# mode setting:
# 0: is full dataset
# 1: full dataset summed in one
# 2: full dataset to 3 sums
mode = 1

tumors = []

cropped_dataset = pre.crop_LUNG_dataset(small_internal, 500)
plt.imshow(cropped_dataset[3])
plt.show()

plt.imshow(cropped_dataset[100])

if mode == 0:
    for cmap in featured_cmaps:
        pre.print_CT_layers_in_table(0, 3, cropped_dataset, cmap)

    for pic in cropped_dataset:
        pre.segment_frame_plot(tumors, pic, 50, 500, 15)


elif mode == 1:
    sum_pic = pre.sum_pics(cropped_dataset)

    pre.segment_frame_plot(tumors, sum_pic, 50, 500, 30)
    plot_all = True
    # tumor.plot_all_sus(tumors, plot_all)
    MASK = True
    #tumor.plot_all(tumors,MASK, croppedOne2 )

elif mode == 2:
    first = cropped_dataset[1:3]
    middle = cropped_dataset[3:5]
    last = cropped_dataset[5:7]

    sum = []
    sum.append(pre.sum_pics(first))
    sum.append(pre.sum_pics(middle))
    sum.append(pre.sum_pics(last))
    print("sum size is: {}".format(len(sum)))

    for pic in sum:
        pre.segment_frame_plot(tumors, pic, 50, 500, 15)

    # plotting all, if false, it will plot only the longer tumors
    plot_all = True
    tumor.plot_all_sus(tumors, plot_all)

print("Found {} suspicious forms:".format(len(tumors)))
