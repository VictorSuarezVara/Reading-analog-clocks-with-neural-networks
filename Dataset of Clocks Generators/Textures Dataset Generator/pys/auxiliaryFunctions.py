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

    dir_tipus_maneta_hores = random.choice(generics["tipus de manetes dhores"])
    hores = Image.open(dir_tipus_maneta_hores).convert('L')
    minuts = Image.open("../Images/minuts.png").convert('L')
    segons = Image.open("../Images/segons.png").convert('L')

    # Calculem l'angle de la hora + rotate
    angle = -hora * 30 + -min * 0.5 + -sec * 0.00833
    black = (0)
    act_hores = hores.rotate(angle, fillcolor=black)

    # Calculem l'angle del minut + rotate
    angle = -min * 6 + -sec * 0.1
    black = (0)
    act_mins = minuts.rotate(angle, fillcolor=black)

    # Calculem l'angle dels secs + rotate
    angle = -sec * 6
    black = (0)
    act_segons = segons.rotate(angle, fillcolor=black)

    # Selecionem un background random per la circumferència
    back_clock_dir = random.choice(generics["textures"])
    back_clock = Image.open(back_clock_dir).resize(hores.size)

    # Seleccionem un background random per las manetes
    back_manetes_dir = random.choice(generics["textures"])
    while back_manetes_dir == back_clock_dir:
        back_manetes_dir = random.choice(generics["textures"])

    back_manetes = Image.open(back_manetes_dir).resize(hores.size)



    # Fem els pastes per juntar-ho tot
    clock = generics["circumferencia"].copy()
    clock.paste(back_clock, generics["circumf_mask"])

    if random.randint(0, 2) == 1:
        # Seleccionem un background random per el voltant de la circumferència
        back_voltant_circumf_dir = random.choice(generics["textures"])
        back_voltant_circumf = Image.open(back_voltant_circumf_dir).resize(generics["size"])
        clock.paste(back_voltant_circumf, generics["voltantCircumferencia"])

    clock.paste(back_manetes, act_hores)

    clock.paste(back_manetes, act_mins)

    if random.randint(0, 2) == 1:
        clock.paste(back_manetes, act_segons)

    clock.paste(generics["black_image"], generics["puntcentric"])

    return clock


def compose_images(foreground, mask_generic, backgrounds, count):
    #https://www.immersivelimit.com/tutorials/composing-images-with-python-for-synthetic-datasets

    # Chose a random backgorund
    background_path = random.choice(backgrounds)

    # Make sure the background path is valid and open the image
    assert os.path.exists(background_path), 'image path does not exist: {}'.format(background_path)
    background = Image.open(background_path)
    background = background.convert('RGB')
    newsize = (512, 512)
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
    hola = 0
