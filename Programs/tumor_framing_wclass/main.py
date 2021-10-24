import pydicom as dicom
import matplotlib.pylab as plt

import os
import numpy as np
import preprocessing as pre
import tumor

FOLDER_PATH = "E:/Egyetem/AI/_Orvosi képfeldolgozás/Datasets/pos_lung_CT_10/tudodaganat/"
featured_cmaps = ["bone","hot","twilight","PuBuGn","inferno","seismic","hsv","twilight_shifted","spring","Accent","bwr","afmhot"  ]


CT_dicom = pre.load_CT(FOLDER_PATH)
CT_kepsorozat = pre.get_pixels_hu(CT_dicom)
small_internal = pre.get_internal_structures(CT_kepsorozat)

cropped_dataset = pre.crop_LUNG_dataset(small_internal, 500)
sum_pic = pre.sum_pics(cropped_dataset)
#pre.segment_frame_plot(sum_pic,50, 500, 15)

tumors = []

for pic in small_internal:
    pre.segment_frame_plot(tumors, pic, 50, 500, 15)

#pre.segment_frame_plot(tumors, sum_pic, 50, 500, 20)

print("Found suspicious shit:", len(tumors))

#tumors[0].plot_Tumor()
tumors[0].plot_onlyTumor(0)

tumor.plot_all_sus(tumors)
