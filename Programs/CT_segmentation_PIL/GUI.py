import tkinter as tk
from tkinter import ttk
import cv2
import PIL.Image, PIL.ImageTk
from matplotlib import cm
import numpy as np
from matplotlib import pyplot
from PIL import Image,ImageTk
import matplotlib.pylab as plt

import img_processing


class GUI_Panel:

    messageBox = tk.Text
    root = tk.Tk()
    imageCanvas = tk.Canvas(root, width=512, height=512)
    imgLabel = ttk.Label()
    orig_dataset = 0
    counter = 0
    cmapNum = 0

    def __init__(self, dataset):
        def get_current_value():
            return int(current_value.get())

        def changeCmap():
            self.cmapNum += 1
            self.messageBox.insert(tk.END, featured_cmaps[self.cmapNum])

        def slider_changed(event):
            value_label.configure(text=get_current_value())
            print("dataset state is:", self.orig_dataset)

            if self.orig_dataset == 0:
                #eredeti dataset:
                img_base = cv2.normalize(src=dataset[get_current_value()], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)

                photo = PIL.ImageTk.PhotoImage(
                    image=PIL.Image.fromarray(img_base)
                )

            elif self.orig_dataset == 1:
                img_internal = cv2.normalize(src=internal[get_current_value()], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
                photo = PIL.ImageTk.PhotoImage(
                    image=PIL.Image.fromarray(img_internal)
                )
            else:
                img_internal = cv2.normalize(src=colored_internal[get_current_value()], dst=None, alpha=0, beta=255,
                                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                photo = PIL.ImageTk.PhotoImage(
                    image=PIL.Image.fromarray(img_internal)
                )

            self.imgLabel.config(image=photo)
            self.imgLabel.photo_ref = photo  # keep a reference

        def changeView():
            self.counter += 1
            self.orig_dataset = self.counter%3
            print("dataset state is:", self.orig_dataset)
            slider_changed(3)

        featured_cmaps = ["bone", "hot", "twilight", "PuBuGn", "inferno", "seismic", "hsv", "twilight_shifted",
                          "spring", "Accent", "bwr", "afmhot"]
        internal = img_processing.get_internal_structures(dataset)
        print("Internal dataset is ready")
        #colored_internal = img_processing.colored_structures(internal)

        colored_internal = []
        for i in range(307):
            colored_internal.append((img_processing.get_colored_img(internal[i])))

        print("Colored internal dataset is ready too")



        #img_processing.make_a_GIF(internal,"internal_strcutres")
        im = cv2.normalize(src=dataset[0], dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.root.geometry('800x800')
        self.root.resizable(True, True)
        self.root.title('Image segmentation')

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=3)

        # slider current value
        current_value = tk.DoubleVar()

        slider_label = ttk.Label(
            self.root,
            text='Slider:',
        )

        slider_label.grid(
            column=0,
            row=0,
            sticky='w'
        )

        slider = ttk.Scale(
            self.root,
            from_=0,
            to=300,
            orient='horizontal',  # vertical
            command=slider_changed,
            variable=current_value
        )

        slider.grid(
            column=1,
            row=0,
            sticky='we',
            padx=50,
            pady=20
        )

        # current value label
        current_value_label = ttk.Label(
            self.root,
            text='Current Value:'
        )

        current_value_label.grid(
            row=1,
            columnspan=1,
            sticky='ne',
            ipadx=10,
            ipady=10
        )


        # value label
        value_label = ttk.Label(
            self.root,
            text=get_current_value()

        )
        value_label.grid(
            row=1,
            columnspan=2,
            sticky='n',
            ipadx=10,
            ipady=10
        )

        ##      img processing:         ##

        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        #photo2 = ImageTk.PhotoImage(Image.open("rs.jpg"))
        photo2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(im))
        self.imgLabel.config(image=photo2)
        self.imgLabel.photo_ref = photo2  # keep a reference

        imgl = ttk.Label(self.root, image=photo2)
        imgl.grid(
            row=3,
            columnspan=3,
            sticky='w',
            padx=20,
            pady=20
        )

        self.imgLabel = imgl

        # Add a PhotoImage to the Canvas
    ##      messagebox init:         ##
        Output = tk.Text(self.root, height=5,
                      width=25,
                      bg="light gray")

        self.messageBox = Output

        self.messageBox.grid(
            row=3,
            columnspan=6,
            sticky='ne',
            padx=20,
            pady=20
        )

        viewButton = tk.Button(text="Change view", command=changeView)

        viewButton.grid(
            row=4,
            columnspan=2,
            sticky='s',
            padx=20,
            pady=20
        )

        changeButton = tk.Button(text="Change cmap", command=changeCmap)

        changeButton.grid(
            row=5,
            columnspan=4,
            sticky='s',
            padx=20,
            pady=20
        )

        ColorButton = tk.Button(text="Internal colored", command=changeCmap)

        changeButton.grid(
            row=5,
            columnspan=4,
            sticky='s',
            padx=20,
            pady=20
        )

        self.root.configure(background='gray')

    def log(self,text):
        self.messageBox.insert(tk.END, text)



