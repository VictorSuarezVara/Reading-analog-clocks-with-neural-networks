import cv2
import pandas as pd
import auxiliaryFunctions as auxF
import os
import random
from matplotlib import pyplot as plt
import sys
from PIL import Image
'''
 # What size you want the datset to be?
'''
wanted_size = 50000

#HiperParams
count = 0

# Get lists of background image paths
backgrounds_dir = os.path.join('../Backgrounds/good_ones')
backgrounds = [os.path.join(backgrounds_dir, file_name) for file_name in os.listdir(backgrounds_dir)]

# Creem els generics
generics = {}
generics["circumferencia"] = Image.open("../Images/circumferencia.png")
generics["size"] = generics["circumferencia"].size

mask_generic = Image.open("../Images/mask_circumferencia.png")


# Inicialitzem el temps
hora = 0
min = 0
sec = 0

first_time = True
while count < wanted_size:

  # Seleccionem la hora, minut i second desitjat
  hora, min, sec = auxF.nextTime(hora, min, sec)

  clock = auxF.rotate(hora, min, sec, generics)

  # Fem el paste, el composite i guardem a disk
  auxF.compose_images(clock, mask_generic, backgrounds, count)


  # Coses del CSV
  if first_time:
    #Create time csv container
    time = pd.DataFrame([[hora, min, sec]], columns=['hour', 'minute', 'second'])
    first_time = False
  else:
    time_aux = pd.DataFrame([[hora, min, sec]], columns=['hour', 'minute', 'second'])
    time = time.append(time_aux, ignore_index=True)

  # Misatge pel creador
  if count % 10000 == 0 and count != 0:
    print("hello, just advising you that ", count, " images had been done")
  count += 1



time.to_csv('../Dataset/label.csv', index=False)
#########################################################33


print("The end of the program of Neural CLock Dataset passed away")