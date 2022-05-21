from PIL import Image
import numpy

im = Image.open("tutorial/result.gif")
im = im.convert("P")

print(im.histogram())

im2 = Image.new("P", im.size, 255)
allpix_im = numpy.asarray(im)
allpix_im2 = numpy.asarray(im2)

# im = im.convert("P")

temp = {}

for x in range(im.size[1]):
    for y in range(im.size[0]):
        pix = im.getpixel((y, x))
        temp[pix] = pix
        if pix > 90 and pix < 255:  # these are the numbers to get_
            im2.putpixel((y, x), 0)
im2.save("output.gif")


inletter = False
foundletter=False
start = 0
end = 0

letters = []

for y in range(im2.size[0]): # slice across_
    for x in range(im2.size[1]): # slice down_
        pix = im2.getpixel((y,x))
        if pix != 255:
            inletter = True

    if foundletter == False and inletter == True:
        foundletter = True
        start = y

    if foundletter == True and inletter == False:
        foundletter = False
        end = y
        letters.append((start,end))

    inletter=False
print(letters)
