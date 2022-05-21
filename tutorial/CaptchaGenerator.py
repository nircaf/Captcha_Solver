# Import the captcha modules
from captcha.image import ImageCaptcha
import random
import os
# Create an image instance of the gicen size
image = ImageCaptcha(width = 300, height = 50)
os.getcwd()
Captcha_length = 6

def generate_captcha(Captcha_length):
    # Image captcha text
    characters = '0123456789abcdefghijklmnopqrstuvwxyz'
    text = []
    for i in range(Captcha_length):
        text.append(characters[random.randint(0,len(characters)-1)])

    text = ''.join(text)

    # generate the image of the given text
    data = image.generate(text)

    # write the image on the given file and save it

    image.write(text, 'data/' + text + '.png')

for j in range(10000):
    generate_captcha(Captcha_length)