import pandas as pd
import os
import random
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import cv2
def nextTime(hora, min, sec):

    # ADD SECONDS
    # sec += 15
    sec += random.randint(15, 45)
    if sec >= 60:
        sec -= 60
        min += 1

    # RECONSTRUCT MINUTES
    if min >= 60:
        min -= 60
        hora += 1

    # RECONSTRUCT HOUR
    if hora >= 12:
        hora -= 12

    return hora, min, sec


def rotate(hora, min, sec, generics):

    hores = Image.open("../Images/hores.jpg").convert('L')
    minuts = Image.open("../Images/mins.jpg").convert('L')
    segons = Image.open("../Images/segons.jpg").convert('L')
    mask_hores = Image.open("../Images/mask_hores.jpg").convert('L')
    mask_minuts = Image.open("../Images/mask_mins.jpg").convert('L')
    mask_segons = Image.open("../Images/mask_segons.jpg").convert('L')


    # Calculem l'angle de la hora + rotate
    angle = -hora * 30 + -min * 0.5 + -sec * 0.00833
    black = (0)
    act_hores = hores.rotate(angle, fillcolor=black)
    act_mask_hores = mask_hores.rotate(angle, fillcolor=black)

    # Calculem l'angle del minut + rotate
    angle = -min * 6 + -sec * 0.1
    black = (0)
    act_mins = minuts.rotate(angle, fillcolor=black)
    act_mask_mins = mask_minuts.rotate(angle, fillcolor=black)

    # Calculem l'angle dels secs + rotate
    angle = -sec * 6
    black = (0)
    act_segons = segons.rotate(angle, fillcolor=black)
    act_mask_segons = mask_segons.rotate(angle, fillcolor=black)

    # Fem els pastes per juntar-ho tot
    clock = generics["circumferencia"].copy()

    clock.paste(act_hores, act_mask_hores)
    clock.paste(act_mins, act_mask_mins)
    clock.paste(act_segons, act_mask_segons)

    return clock


def compose_images(foreground, mask_generic, backgrounds, count):
    #https://www.immersivelimit.com/tutorials/composing-images-with-python-for-synthetic-datasets

    # Chose a random backgorund
    background_path = random.choice(backgrounds)

    # Make sure the background path is valid and open the image
    assert os.path.exists(background_path), 'image path does not exist: {}'.format(background_path)
    background = Image.open(background_path)
    background = background.convert('RGB')
    newsize = (4000, 4000)
    background = background.resize(newsize)


    # Scale the foreground
    scale = random.random() * .5 + .6  # Pick something between .5 and 1
    new_size = (int(foreground.size[0] * scale), int(foreground.size[1] * scale))
    foreground = foreground.resize(new_size, resample=Image.BICUBIC)

    #Open mask and scaleit too
    mask_generic = mask_generic.resize(new_size, resample=Image.BICUBIC).convert("L")

    # Choose a random x,y position for the foreground
    max_xy_position = (background.size[0] - foreground.size[0], background.size[1] - foreground.size[1])
    if max_xy_position[0] < 1:
        hola = 0

    if max_xy_position[1] < 1:
        hola = 2


    paste_position = (random.randint(0, max_xy_position[0]), random.randint(0, max_xy_position[1]))

    # Extract the alpha channel from the foreground and paste it into a new image the size of the background
    composition = background.copy()
    composition.paste(foreground, paste_position, mask_generic)
    composition = composition.resize((256, 256))
    composition = composition.convert('L')

    composite_path = os.path.join("../Dataset/images/" + str(count) + ".jpg")
    composition.save(composite_path, 'JPEG')


def openImage(path, asgreey=False, showImg=False, showDims=False):
    image = Image.open(path).convert('RGB')

    if asgreey:
        image = ImageOps.grayscale(image)

    width, height = image.width, image.height
    if showImg:
        image.show()

    if showDims:
        print(width, height)

    return image


def change_pil_to_Cv2(im, showDims=False):
    im = np.array(im)
    im.astype(np.float32)
    if showDims:
        print(im.shape)
    return im


def create_a_mask(dir):
    mask = openImage(dir, asgreey=True)
    mask = change_pil_to_Cv2(mask)

    mask = (mask < 230) * 255

    plt.imsave(dir, mask, cmap='Greys')
