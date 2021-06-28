import cv2
import pandas as pd
import auxiliaryFunctions as auxF
import os
import random
from matplotlib import pyplot as plt
import sys


'''
 # Here you can chose the clocks that you want in your dataset and the quantity of data in total
 0 = video 0
 1 = video 1
 2 = video 2
 3 = video 3
  
'''
list_of_datsets = [0, 0, 0, 0, 1, 2, 2, 3, 3]


'''
 # What size you want the datset to be?
'''
wanted_size = 50000

#HiperParams do not touch better
first_time = True
count = 0

#Dir of vídeos
videos_dir = os.path.join('../Images & videos/Videos clocks/clock_video_')
len_videos = len(os.listdir("../Images & videos/Videos clocks"))

#List of masks
masks_dir = os.path.join('../Images & videos/Masks clocks/mask_clock_video_')

# Get lists of background image paths
#http://web.mit.edu/torralba/www/indoor.html
backgrounds_dir = os.path.join('../Backgrounds/good_ones')
backgrounds = [os.path.join(backgrounds_dir, file_name) for file_name in os.listdir(backgrounds_dir)]



while count < wanted_size:
  first_frame = True
  video_selected = random.choice(list_of_datsets)

  video_root = videos_dir + str(video_selected) + ".mp4"
  mask_root = masks_dir + str(video_selected) + ".jpg"

  #This funct help u create a mask better
  # auxF.create_a_mask(mask_root)

  # Start capturing frames of a video
  vidcap = cv2.VideoCapture(video_root)

  hour, minute, second, addition_minutes, addition_seconds = auxF.videos_and_first_frame_info(video_root)

  if first_time:
    #Create time csv container
    time = pd.DataFrame([[hour, minute, second]], columns=['hour', 'minute', 'second'])

  if video_selected != 0:
    success, image = vidcap.read()

  add_secs = 0
  while 1:
    success, image = vidcap.read()
    if not success:
      break


    if not first_frame:

      #ADD SECONDS
      second += addition_seconds[add_secs]
      if len(addition_seconds)==add_secs+1:
        add_secs = 0
      else:
        add_secs += 1

      if second >= 60:
        second -= 60
        minute += 1

      #ADD MINUTES
      minute += addition_minutes
      if minute >= 60:
        minute -= 60
        hour += 1

      #ADD HOUR
      if hour >= 12:
        hour -= 12

    else:
      first_frame = False


    if not first_time:

      # Write on CSV the content of the frames
      time_aux = pd.DataFrame([[hour, minute, second]], columns=['hour', 'minute', 'second'])
      time = time.append(time_aux, ignore_index=True)

    else:
      first_time = False

    #Paste and write on disk
    background_path = random.choice(backgrounds)
    composite = auxF.compose_images(image, mask_root, background_path)

    composite_path = os.path.join("../Dataset/images/" + str(count) + ".jpg")
    composite.save(composite_path, 'JPEG')



    if count % 10000 == 0 and count != 0:
      print("hello, just advising you that ", count, " images had been done")
    count += 1

    if count == wanted_size:
      break


time.to_csv('../Dataset/label.csv', index=False)


#########################################################33


print("The end of the program passed away, :( ")