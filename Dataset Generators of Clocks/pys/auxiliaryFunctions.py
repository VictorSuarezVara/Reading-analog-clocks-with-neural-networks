import pandas as pd
import os
import random
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import cv2

def videos_and_first_frame_info(video_root):

    if video_root == '../Images & videos/Videos clocks/clock_video_0.mp4':  # datsize = 336
        hour = 4
        minute = 17
        second = 26
        addition_minutes = 2
        addition_seconds = [8]

    elif video_root == '../Images & videos/Videos clocks/clock_video_1.mp4':    # datsize = 1294
        hour = 0
        minute = 1
        second = 57
        addition_minutes = 0
        addition_seconds = [30, 40, 30]

    elif video_root == '../Images & videos/Videos clocks/clock_video_2.mp4':    # datsize = 647
        hour = 0
        minute = 7
        second = 26
        addition_minutes = 1
        addition_seconds = [6.73]

    elif video_root == '../Images & videos/Videos clocks/clock_video_3.mp4':     # datsize = 595
        hour = 0
        minute = 1
        second = 15
        addition_minutes = 1
        addition_seconds = [12.44]

    elif video_root == '../Images & videos/Videos clocks/clock_video_4.mp4':    # datsize = 517
        hour = 1
        minute = 17
        second = 9
        addition_minutes = 1
        addition_seconds = [23.3]



    return hour, minute, second, addition_minutes, addition_seconds


def compose_images(foreground, mask_root, background_path):
    #https://www.immersivelimit.com/tutorials/composing-images-with-python-for-synthetic-datasets

    # Make sure the background path is valid and open the image
    assert os.path.exists(background_path), 'image path does not exist: {}'.format(background_path)
    background = Image.open(background_path)
    background = background.convert('RGB')
    newsize = (512, 512)
    background = background.resize(newsize)

    # Change to PIL foreground
    newsize = (450, 300)
    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    foreground = Image.fromarray(foreground,"RGB")
    foreground = foreground.resize(newsize)

    # Scale the foreground
    scale = random.random() * .5 + .6  # Pick something between .5 and 1
    new_size = (int(foreground.size[0] * scale), int(foreground.size[1] * scale))
    foreground = foreground.resize(new_size, resample=Image.BICUBIC)

    # Add any other transformations here...


    #Open mask and scaleit too
    mask_foreground = openImage(mask_root, asgreey=True)
    mask_foreground = mask_foreground.resize(new_size, resample=Image.BICUBIC)


    # Choose a random x,y position for the foreground
    max_xy_position = (background.size[0] - foreground.size[0], background.size[1] - foreground.size[1])
    if max_xy_position[0] < 1:
        hola = 0

    if max_xy_position[1] < 1:
        hola = 2


    paste_position = (random.randint(0, max_xy_position[0]), random.randint(0, max_xy_position[1]))


    # Extract the alpha channel from the foreground and paste it into a new image the size of the background
    composition = background.copy()
    composition.paste(foreground, paste_position, mask_foreground)
    #composition.show()

    return composition


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


def erase_whater_marks():
    '''
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2

    # extra
    import numpy as np
    import cv2

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture('../Images & videos/prueba.mp4')

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            for x in range(frame.shape[0]):
                for y in range(frame.shape[1]):
                    if y > 230 and y < 412 and x > 134 and x < 220:
                        if frame[x][y][0] > 70 and frame[x][y][0] < 110:
                            frame[x][y][0] = 50
                            frame[x][y][1] = 50
                            frame[x][y][2] = 50

                        elif frame[x][y][0] > 120:
                            num = 4 + x / 75
                            frame[x][y][0] = 187
                            frame[x][y][1] = 185
                            frame[x][y][2] = 181

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
    '''
    HOLA = 1