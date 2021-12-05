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

# LUNG_TRESH = -600
LUNG_TRESH = -600


def load_CT(PATH):
    slices = [dicom.dcmread(PATH + '/' + s) for s in os.listdir(PATH)]
    slices = sorted(slices, key=lambda s: s.SliceLocation)
    slices = slices[::-1]
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

    # resize image TODO do i need resize?:
    # resized = cv2.resize(crop, dim, interpolation=cv2.INTER_CUBIC)
    return crop


def crop_rgb_LUNG(image, SCALE, minr, minc, maxr, maxc):
    padding = 20
    # minr, minc, maxr, maxc = get_cropping_size(image, padding)
    crop = image[minr:minr + maxr - minr, minc:minc + maxc - minc]
    scale_percent = 220  # percent of original size
    width = int(crop.shape[1] * SCALE / 100)
    height = int(crop.shape[0] * SCALE / 100)
    dim = (height, width)

    # rgb_resized = resize(crop, dim)
    return crop


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
    framearea = getCircleArea(a, b)
    # true, if the width and height values are close enough (TRESHOLD) AND the area
    if abs(a - b) < TRESHOLD and\
        framearea * AREA_TRESHOLD_PERCENT < REGION_AREA:
        return True
    else:
        return False


def segment_frame_plot(tumors, image, base_image, MINSIZE, MAXSIZE, PADDING, PLOTTING_ENABLED, imageNUM):
    # tresholds:

    # originally:
    # FRAMING_TRESHOLD = 5
    # AREA_TRESHOLD_PERCENTAGE = 0.40
    FRAMING_TRESHOLD = 5
    AREA_TRESHOLD_PERCENTAGE = 0.40

    is_all_zero = np.all((image == 0))
    if not is_all_zero:
        thresh = threshold_otsu(image)

        #hope it wont causes any problems in the future
        #bw = closing(image > thresh, square(2))
        #bw = closing(image > thresh, square(2))
        cntr = 0
        # remove artifacts connected to image border
        #cleared = clear_border(image)

        # label image regions
        label_image = label(image)
        # to make the background transparent, pass the value of `bg_label`,
        # and leave `bg_color` as `None` and `kind` as `overlay`
        image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
        if (PLOTTING_ENABLED):
            fig, ax = plt.subplots(figsize=(10, 6), ncols=2)
            # dest = cv2.addWeighted(image, 0.5, base_image, 0.5, 0.0)
            dest = base_image * 0.95 + image * (1.0 - 0.95)
            dest2 = base_image * 0.9 + image * (1.0 - 0.9)

            ax[0].imshow(image, interpolation="lanczos",  cmap="afmhot")
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
                                          fill='red', edgecolor='red', facecolor='red', linewidth=0.5)
                    # print("smallerarea: {}, REGION_AREA{}".format(pow(min(maxc - minc, maxr - minr), 2) * math.pi,region.area))
                    cntr += 1
                    x = minc + (maxc - minc) / 2
                    y = minr + (maxr - minr) / 2

                    tmp_tumor = Tumor(framing, image, region.area,circle_area, x,y)
                    tmp_tumor.setStartimg(imageNUM)
                    length = findTumor(tumors, tmp_tumor)
                    circle2 = mpatches.Circle((minc + (maxc - minc) / 2, minr + (maxr - minr) / 2),
                                              radius * 3,
                                              fill=False, ec=(1, 1, 0, 1), linewidth=length * 0.3)

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
                            (minc + radius * 0.9, minr + radius * 0.5),
                            color='cyan', weight='bold',
                            fontsize=5, ha='left', va='center')
                        #print("#{} Fill rate={}%, area= {}, ({},{}), radius is:{}".format(cntr,round(region.area / circle_area * 100,2),region.area, x, y, radius))

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
    fig.show()
    # fig.write_html("LUNG.html")
    plt.show()


def plotly_3d(image):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the
    # camera
    p = image.transpose(2, 1, 0)

    verts, faces, _, _ = measure.marching_cubes_lewiner(p)

    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    """
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    """
    import plotly.graph_objects as go
    import pandas as pd

    z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')
    fig = go.Figure(data=[go.Surface(z=z_data.values)])

    # fig = go.Figure(data=[go.Mesh3d(z=p, color='rgb(255,0,0)')])

    fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))

    fig.show()


""" 
import mayavi
    from mayavi import mlab
    mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    surH = mlab.surf(image, warp_scale='auto', mask=image.mask)
    mlab.show()
"""


def plotly_probe(image):
    import plotly.graph_objects as go
    import numpy as np

    z1 = np.array([
        [8.83, 8.89, 8.81, 8.87, 8.9, 8.87],
        [8.89, 8.94, 8.85, 8.94, 8.96, 8.92],
        [8.84, 8.9, 8.82, 8.92, 8.93, 8.91],
        [8.79, 8.85, 8.79, 8.9, 8.94, 8.92],
        [8.79, 8.88, 8.81, 8.9, 8.95, 8.92],
        [8.8, 8.82, 8.78, 8.91, 8.94, 8.92],
        [8.75, 8.78, 8.77, 8.91, 8.95, 8.92],
        [8.8, 8.8, 8.77, 8.91, 8.95, 8.94],
        [8.74, 8.81, 8.76, 8.93, 8.98, 8.99],
        [8.89, 8.99, 8.92, 9.1, 9.13, 9.11],
        [8.97, 8.97, 8.91, 9.09, 9.11, 9.11],
        [9.04, 9.08, 9.05, 9.25, 9.28, 9.27],
        [9, 9.01, 9, 9.2, 9.23, 9.2],
        [8.99, 8.99, 8.98, 9.18, 9.2, 9.19],
        [8.93, 8.97, 8.97, 9.18, 9.2, 9.18]
    ])

    z2 = z1 + 1
    z3 = z1 - 1

    fig = go.Figure(data=[
        go.Surface(z=z1),
        go.Surface(z=z2, showscale=False, opacity=0.9),
        go.Surface(z=z3, showscale=False, opacity=0.9)

    ])

    fig.show()


def slicer(image):
    # Import data
    import time
    import numpy as np

    from skimage import io

    vol = io.imread("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif")
    volume = vol.T
    r, c = volume[0].shape

    # Define frames
    import plotly.graph_objects as go
    nb_frames = 68

    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=(6.7 - k * 0.1) * np.ones((r, c)),
        surfacecolor=np.flipud(volume[67 - k]),
        cmin=0, cmax=200
    ),
        name=str(k)  # you need to name the frame for the animation to behave properly
    )
        for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=6.7 * np.ones((r, c)),
        surfacecolor=np.flipud(volume[67]),
        colorscale='Gray',
        cmin=0, cmax=200,
        colorbar=dict(thickness=20, ticklen=4)
    ))

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title='Slices in volumetric data',
        width=600,
        height=600,
        scene=dict(
            zaxis=dict(range=[-0.1, 6.8], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders
    )

    fig.show()


import imageio
from IPython import display


def make_a_GIF(imgs, GIFNAME):
    imageio.mimsave(f'./{GIFNAME}.gif', imgs, duration=0.1)
    display.Image(f'./{GIFNAME}.gif', format='png')


def plotly_img(dataset):
    fig = px.imshow(dataset, animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
    fig.show()


def cleared_tum(dataset):
    new = []
    for image in dataset:
        thresh = threshold_otsu(image)

        bw = closing(image > thresh, square(3))
        cntr = 0
        # remove artifacts connected to image border
        cleared = clear_border(bw)

        # label image regions
        label_image = label(cleared)
        new.append(label_image)
    return new
