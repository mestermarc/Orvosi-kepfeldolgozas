
import pydicom as dicom
import matplotlib.pylab as plt

import os
import numpy as np
# segmentation:

from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage.morphology import ball, binary_closing, closing, square
from skimage.measure import label, regionprops
import cv2
import scipy
from skimage.segmentation import clear_border


def load_CT(PATH):
    slices = [dicom.dcmread(PATH + '/' + s) for s in os.listdir(PATH)]
    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def print_CT_layers(first,last,dataset,CMAP):
    for i in range(first,last):
        plt.imshow(dataset[i],cmap=CMAP)
        plt.title(str(i)+". picture with cmap:"+CMAP)
        plt.show()
       # plt.close(fig)


def print_CT_layers_in_table(first, last, dataset, CMAP):
    rows = 1
    columns = 4
    # fig, ax_lst = plt.subplots(2, 2, figsize=(12,8))
    fig = plt.figure(figsize=(30, 30))
    for i in range(first, last):
        if ((i - first) % 4 == 0):
            rows = rows + 1;
            fig = plt.figure(figsize=(30, 30))
        fig.add_subplot(1, columns, ((i - first) % 4) + 1)
        plt.axis('off')
        plt.imshow(dataset[i], cmap=CMAP)
        plt.title(str(i) + ".")
    plt.show()


import imageio
from IPython import display
def make_a_GIF(imgs, GIFNAME):
    data = 255 * imgs  # Now scale by 255
    img = data.astype(np.uint8)
    imageio.mimsave(f'./{GIFNAME}.gif', img, duration=0.1)
    display.Image(f'./{GIFNAME}.gif', format='png')


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image >= -700, dtype=np.int8) + 1
    labels = measure.label(binary_image)
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to
    # something like morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets inside body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image


def total_lung_MASK(img):
    return 1 - scipy.ndimage.filters.gaussian_filter(1 - img, 0.7, order=0, output=None, mode='reflect', cval=0.7, truncate=7.0)


def get_internal_structures(dataset):
    segmented_lung = segment_lung_mask(dataset, fill_lung_structures=False)
    return total_lung_MASK(segmented_lung)-segmented_lung

def get_internal_structures_in_PIL(dataset):
    segmented_lung = segment_lung_mask(dataset, fill_lung_structures=False)
    return cv2.normalize(src=total_lung_MASK(segmented_lung)-segmented_lung, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def preprocessing(CT_kepsorozat):
    segmented_lungs = segment_lung_mask(CT_kepsorozat, fill_lung_structures=False)
    segmented_lungs_fill = segment_lung_mask(CT_kepsorozat, fill_lung_structures=True)
    internal_structures = segmented_lungs_fill - segmented_lungs
    # own internal structures:
    lung_mask = total_lung_MASK(segmented_lungs_fill)
    own_internal_structures = lung_mask - segmented_lungs
    return own_internal_structures


def get_colored_img(image):
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
    return image_label_overlay


def colored_structures(dataset):
    datas = []
    for i in range(len(dataset)):
        datas.append((get_colored_img(dataset[i]) * 255).astype(np.uint8))
    return datas
