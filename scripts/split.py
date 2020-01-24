import random
import glob


images = glob.glob("images/*.jpg")
images = ["laser/" + image for image in images]
random.shuffle(images)

train = open("train.txt", "w")
test = open("test.txt", "w")

for i in range(len(images)):
    if i < 0.85 * len(images):
        train.write(images[i] + "\n")
    else:
        test.write(images[i] + "\n")