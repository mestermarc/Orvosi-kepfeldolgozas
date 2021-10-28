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

small_internal = pre.get_internal_structures(CT_kepsorozat)

print("dataset size is: {}".format(len(small_internal)))

# mode setting:
# 0: is full dataset
# 1: full dataset summed in one
# 2: full dataset to 3 sums
mode = 0

tumors = []

cropped_dataset = pre.crop_LUNG_dataset(small_internal, 500)
internal_dataset = pre.cutoffborder(cropped_dataset)

if mode == 0:
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
