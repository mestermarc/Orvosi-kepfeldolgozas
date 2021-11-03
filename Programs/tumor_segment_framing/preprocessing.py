import math
import os

import cv2
import matplotlib.patches as mpatches
import matplotlib.pylab as plt
import numpy as np
import plotly.express as px
import pydicom as dicom
import scipy
import skimage.segmentation as seg
# segmentation:
from skimage import measure
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
# segmentation:
from skimage.transform import resize

from tumor import Tumor, findTumor

LUNG_TRESH = -600


def load_CT(PATH):
    slices = [dicom.dcmread(PATH + '/' + s) for s in os.listdir(PATH)]
    slices = sorted(slices, key=lambda s: s.SliceLocation)
    # kellett a reverse
    sli = slices[::-1]
    return sli


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
    binary_image = np.array(image >= LUNG_TRESH, dtype=np.int8) + 1
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


def cutoffborder(dataset):
    cropped = []
    for data in dataset:
        w = seg.flood_fill(1 - data, (1, 1), 0)
        w = seg.flood_fill(w, (1, 1), 1)
        cropped.append(1 - w)
    return cropped


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

    # resize image TODO interpolation changes:
    resized = cv2.resize(crop, dim, interpolation=cv2.INTER_CUBIC)
    return resized


def crop_rgb_LUNG(image, SCALE, minr, minc, maxr, maxc):
    padding = 20
    # minr, minc, maxr, maxc = get_cropping_size(image, padding)
    crop = image[minr:minr + maxr - minr, minc:minc + maxc - minc]
    scale_percent = 220  # percent of original size
    width = int(crop.shape[1] * SCALE / 100)
    height = int(crop.shape[0] * SCALE / 100)
    dim = (height, width)

    rgb_resized = resize(crop, dim)
    return rgb_resized


def crop_LUNG_dataset(dataset, SCALE):
    padding = 40
    minr, minc, maxr, maxc = get_cropping_size(dataset[120], padding)

    results = []
    for data in dataset:
        results.append(crop_LUNG(data, SCALE, minr, minc, maxr, maxc))
    return results


def crop_rgb_LUNG_dataset(dataset, SCALE, frame_img):
    padding = 40
    minr, minc, maxr, maxc = get_cropping_size(frame_img[120], padding)

    results = []
    for data in dataset:
        results.append(crop_rgb_LUNG(data, SCALE, minr, minc, maxr, maxc))
    return results


def sum_pics(dataset):
    dst = dataset[0]
    for i in range(1, len(dataset) - 1):
        dst = cv2.addWeighted(dst, 0.5, dataset[i], 0.5, 0.0)
    return dst


def getCircleArea(a, b):
    radius = ((max(a, b)) / 2) * 1.2  # multiplied, to make a little overlay
    return radius * radius * math.pi


def aboutSQ(a, b, REGION_AREA, TRESHOLD, AREA_TRESHOLD_PERCENT):
    # if the frame is "circle" enough:
    # bigcircle_radius
    framearea = getCircleArea(a, b)
    # enters, if the width and height values are close enough (TRESHOLD) AND
    if (abs(a - b) < TRESHOLD and framearea * AREA_TRESHOLD_PERCENT < REGION_AREA):
        # print("Success, bc: framearea: {}, REGION_AREA: {}".format(framearea, REGION_AREA))
        # print("SUCCESS")
        return True
    else:
        return False


def segment_frame_plot(tumors, image, base_image, MINSIZE, MAXSIZE, PADDING, PLOTTING_ENABLED, imageNUM):
    # tresholds:

    # originally:
    FRAMING_TRESHOLD = 100
    AREA_TRESHOLD_PERCENTAGE = 0.40

    is_all_zero = np.all((image == 0))
    if not is_all_zero:
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
        if (PLOTTING_ENABLED):
            fig, ax = plt.subplots(figsize=(10, 6), ncols=2)
            # dest = cv2.addWeighted(image, 0.5, base_image, 0.5, 0.0)
            dest = base_image * 0.95 + image * (1.0 - 0.95)
            dest2 = base_image * 0.9 + image * (1.0 - 0.9)

            ax[0].imshow(dest, cmap="afmhot")
            ax[1].imshow(dest2, cmap="bone")

        for region in regionprops(label_image):
            # take regions with large enough areas
            if region.area >= MINSIZE and region.area < MAXSIZE:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                # if the frame is circle enough:
                if aboutSQ(maxc - minc, maxr - minr, region.area, FRAMING_TRESHOLD, AREA_TRESHOLD_PERCENTAGE):
                    a = maxr - minr
                    b = maxc - minc
                    radius = max(a, b) / 2
                    circle_area = getCircleArea(a, b)
                    smallrect = mpatches.Rectangle((minc, minr), maxc - minc,
                                                   maxr - minr,
                                                   fill=False, ec=(0.5, 1, 0, 0.8), linewidth=1)

                    framing = mpatches.Rectangle((minc - PADDING, minr - PADDING), maxc - minc + PADDING * 2,
                                                 maxr - minr + PADDING * 2,
                                                 fill=False, edgecolor='white', linewidth=1)

                    framing2 = mpatches.Rectangle((minc - PADDING, minr - PADDING), maxc - minc + PADDING * 2,
                                                  maxr - minr + PADDING * 2,
                                                  fill=False, edgecolor='white', linewidth=1)
                    circle = mpatches.Circle((minc + (maxc - minc) / 2, minr + (maxr - minr) / 2),
                                             radius * 3,
                                             fill=False, ec=(1, 0, 0, 1), linewidth=1)

                    dot = mpatches.Circle((minc + (maxc - minc) / 2, minr + (maxr - minr) / 2), 0.9,
                                          fill='black', edgecolor='black', facecolor='black', linewidth=1)
                    # print("smallerarea: {}, REGION_AREA{}".format(pow(min(maxc - minc, maxr - minr), 2) * math.pi,region.area))
                    cntr += 1
                    x = minc + (maxc - minc) / 2
                    y = minr + (maxr - minr) / 2

                    tmp_tumor = Tumor(framing, image, region.area, pow(min(maxc - minc, maxr - minr), 2) * math.pi, x,
                                      y)
                    tmp_tumor.setStartimg(imageNUM)
                    length = findTumor(tumors, tmp_tumor)
                    circle2 = mpatches.Circle((minc + (maxc - minc) / 2, minr + (maxr - minr) / 2),
                                              radius * 3,
                                              fill=False, ec=(1, 1, 0, 1), linewidth=length*1.3)

                    if (PLOTTING_ENABLED):
                        ax[0].add_patch(framing)

                        if length > 0:
                            ax[1].add_patch(circle2)
                            ax[1].annotate(
                                "LEN:{}".format(length),
                                (x, y),
                                color='red', weight='bold',
                                fontsize=5, ha='left', va='center')
                        else:
                            ax[1].add_patch(circle)

                        ax[0].add_patch(dot)
                        # number, frame area, region area, fill rate:
                        ax[0].annotate(
                            "#{} Fill rate={}%, area= {}".format(cntr, round(region.area / circle_area * 100, 2),
                                                                 region.area),
                            (minc + radius*0.9, minr + radius*0.5),
                            color='white', weight='bold',
                            fontsize=5, ha='left', va='center')
                        print("#{} Fill rate={}%, area= {}, ({},{}), radius is:{}".format(cntr,
                                                                            round(region.area / circle_area * 100, 2),
                                                                            region.area, x, y,radius))

        if (PLOTTING_ENABLED):
            ax[0].set_axis_off()
            ax[1].set_axis_off()
            plt.tight_layout()
            plt.show()


from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_3d(image):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the
    # camera
    p = image.transpose(2, 1, 0)

    verts, faces, _, _ = measure.marching_cubes_lewiner(p)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of
    # triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


import imageio
from IPython import display


def make_a_GIF(imgs, GIFNAME):
    imageio.mimsave(f'./{GIFNAME}.gif', imgs, duration=0.1)
    display.Image(f'./{GIFNAME}.gif', format='png')


def plotly_img(dataset):
    fig = px.imshow(dataset, animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
    fig.show()
