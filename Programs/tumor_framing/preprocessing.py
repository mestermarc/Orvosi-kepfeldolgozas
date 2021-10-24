import pydicom as dicom
import matplotlib.pylab as plt
import math

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
import matplotlib.patches as mpatches

# segmentation:
from skimage import measure, morphology
from skimage.morphology import ball, binary_closing
from skimage.measure import label, regionprops
import cv2
import scipy
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.patches as mpatches


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
    return 1 - scipy.ndimage.filters.gaussian_filter(1 - img, 0.7, order=0, output=None, mode='reflect', cval=0.7,
                                                     truncate=7.0)


def get_internal_structures(dataset):
    segmented_lung = segment_lung_mask(dataset, fill_lung_structures=False)
    return total_lung_MASK(segmented_lung) - segmented_lung


def print_CT_layers_in_table(first, last, dataset, CMAP):
    rows = 1
    columns = 4
    # fig, ax_lst = plt.subplots(2, 2, figsize=(12,8))
    fig = plt.figure(figsize=(30, 30))
    for i in range(first, last):
        if ((i - first) % 4 == 0):
            rows = rows + 1
            fig = plt.figure(figsize=(30, 30))
        fig.add_subplot(1, columns, ((i - first) % 4) + 1)
        plt.axis('off')
        plt.imshow(dataset[i], cmap=CMAP)
        plt.title(str(i) + ".")
    plt.show()


def get_cropping_size(image, PADDING):
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)
    print("called")
    # label image regions
    label_image = label(cleared)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area > 500:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            break
    return minr - PADDING, minc - PADDING, maxr + PADDING, maxc + PADDING


def crop_LUNG(image, SCALE, minr, minc, maxr, maxc):
    padding = 20
    # minr, minc, maxr, maxc = get_cropping_size(image, padding)
    crop = image[minr:minr + maxr - minr, minc:minc + maxc - minc]
    scale_percent = 220  # percent of original size
    width = int(crop.shape[1] * SCALE / 100)
    height = int(crop.shape[0] * SCALE / 100)
    dim = (width, height)

    crop = np.array(crop, dtype='uint8')

    # resize image
    resized = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)
    return resized


def crop_LUNG_dataset(dataset, SCALE):
    padding = 20
    minr, minc, maxr, maxc = get_cropping_size(dataset[0], padding)

    results = []
    for data in dataset:
        results.append(crop_LUNG(data, SCALE, minr, minc, maxr, maxc))
    return results


def sum_pics(dataset):
    dst = dataset[0]
    for i in range(1, len(dataset) - 1):
        print("dst = cv2.addWeighted(dst, 0.5, dataset[{}], 0.5, 0.0)".format(i))
        dst = cv2.addWeighted(dst, 0.5, dataset[i], 0.5, 0.0)
    return dst


def aboutSQ(a, b, REGION_AREA, TRESHOLD, AREA_TRESHOLD_PERCENT):
    # if the frame is circle enough:
    smaller = min(a, b)
    smallerarea = pow(smaller / 2, 2) * math.pi
    if (abs(a - b) < TRESHOLD and smallerarea > REGION_AREA * AREA_TRESHOLD_PERCENT):
        print("smallerarea: {}, REGION_AREA{}".format(smallerarea, REGION_AREA))
        print("SUCCESS")
        return True
    else:
        return False


def segment_frame_plot(image, MINSIZE, MAXSIZE, PADDING):
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))
    cntr = 0
    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)

    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= MINSIZE and region.area < MAXSIZE:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            # if the frame is circle enough:
            if aboutSQ(maxc - minc, maxr - minr, region.area, 40, 0.67):
                # if(True):
                rect = mpatches.Rectangle((minc - PADDING, minr - PADDING), maxc - minc + PADDING * 2,
                                          maxr - minr + PADDING * 2,
                                          fill=False, edgecolor='white', linewidth=1)
                circle = mpatches.Circle((minc + (maxc - minc) / 2, minr + (maxr - minr) / 2),
                                         min(maxc - minc, maxr - minr),
                                         fill=False, edgecolor='red', linewidth=1)

                dot = mpatches.Circle((minc + (maxc - minc) / 2, minr + (maxr - minr) / 2), 0.9,
                                      fill='red', edgecolor='red', linewidth=1)
                print("smallerarea: {}, REGION_AREA{}".format(pow(min(maxc - minc, maxr - minr), 2) * math.pi,
                                                              region.area))
                cntr += 1
                ax.add_patch(rect)
                ax.add_patch(circle)
                ax.add_patch(dot)
                # print("x= {}, y = {}, w = {}, h = {}, rectangle size = ".format(minr, minc, maxr, maxc))
                print(cntr)
                ax.annotate("Rectangle{}".format(cntr), (minc - 10, minr - 10), color='white', weight='bold',
                            fontsize=5, ha='left', va='center')

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
