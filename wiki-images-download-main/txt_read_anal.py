import base64
import io
from PIL import Image

def alltext_to_png(title):
    with open('C:\\NeuroBrave\\Scrapy\\wiki-images-download-main\\wiki_images\\spiders\\alltext.txt') as f:
        lines = f.readlines()

    s = 'data:image/jpg;base64'
    matched_indexes = []
    for i,data in enumerate(lines):
        if data.find(s)>0:
            matched_indexes.append(i)
            break

    data2 = data[data.find(s)+len(s)+1:data.find('no-repeat scroll')-3]

    b = bytes(data2, 'utf-8')

    z = b[b.find(b'/9'):]
    im = Image.open(io.BytesIO(base64.b64decode(z))).save(title + '.png')
    print("Converted " )

