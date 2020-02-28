#!/usr/bin/env python

from gimpfu import *
import math

img = gimp.image_list()[0]

for layer in img.layers:
    pdb.plug_in_autostretch_hsv(img, layer)
    layer.visible = False

width, height = 100, 100
opacity = 0
label_layer = gimp.Layer(img, "Label", width, height, RGB_IMAGE, opacity, NORMAL_MODE)

img.add_layer(label_layer, 0)
pdb.gimp_edit_fill(label_layer, BACKGROUND_FILL)
