# import the Python Image processing Library



from PIL import Image
import glob
import os
for count in range(50000):
    
    path = os.path.join("../Dataset512rgb/images/" + str(count) + ".jpg")
    image = Image.open(path)

    image = image.resize((256, 256))

    path2 = os.path.join("../Datasetrgb/images/" + str(count) + ".jpg")
    image.save(path2)



