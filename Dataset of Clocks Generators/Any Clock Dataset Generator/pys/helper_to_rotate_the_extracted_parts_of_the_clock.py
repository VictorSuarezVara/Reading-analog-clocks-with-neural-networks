# import the Python Image processing Library

from PIL import Image

# Obrim les imatges generiques
circumferencia = Image.open("../Images/circumferencia.png").convert('L')
hores = Image.open("../Images/hores.png").convert('L')
minuts = Image.open("../Images/minuts.png").convert('L')
segons = Image.open("../Images/segons.png").convert('L')
mask_hores = Image.open("../Images/mask_hores.png").convert('L')
mask_minuts = Image.open("../Images/mask_minuts.png").convert('L')
mask_segons = Image.open("../Images/mask_segons.png").convert('L')


# Seleccionem la hora, minut i second desitjat
hora = 11
min = 41
sec = 17.56

# Calculem l'angle de la hora + rotate
angle = -hora*30 + -min*0.5 + -sec*0.00833
black = (0)
act_hores = hores.rotate(-angle, fillcolor=black)
act_mask_hores = mask_hores.rotate(-angle, fillcolor=black)
act_hores.save('../Images/new/hores.jpg', quality=100)
act_mask_hores.save('../Images/new/mask_hores.jpg', quality=100)
# act_hores.show()


# Calculem l'angle del minut + rotate
angle = -min*6 + -sec*0.1
black = (0)
act_mins = minuts.rotate(-angle, fillcolor=black)
act_mask_mins = mask_minuts.rotate(-angle, fillcolor=black)
act_mins.save('../Images/new/mins.jpg', quality=100)
act_mask_mins.save('../Images/new/mask_mins.jpg', quality=100)
# act_mins.show()

# Calculem l'angle dels secs + rotate
angle = -sec*6
black = (0)
act_segons = segons.rotate(-angle, fillcolor=black)
act_mask_segons = mask_segons.rotate(-angle, fillcolor=black)
act_segons.save('../Images/new/segons.jpg', quality=100)
act_mask_segons.save('../Images/new/mask_segons.jpg', quality=100)
# act_segons.show()


# Fem els pastes per juntar-ho tot
clock = circumferencia.copy()
clock = clock.convert("L")
clock.paste(act_hores, act_mask_hores)
clock.paste(act_mins, act_mask_mins)
clock.paste(act_segons, act_mask_segons)
clock.show()
# clock.save('data/dst/rocket_pillow_paste_mask_circle.jpg', quality=95)

