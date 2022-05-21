from txt_read_anal import alltext_to_png
import os
import sys
import time
import subprocess
from pathlib import Path

batcmd='cmd /c "cd C:\\NeuroBrave\\Scrapy\\wiki-images-download-main\\wiki_images\\spiders & scrapy crawl paris"'

dir_data = 'C:\\NeuroBrave\\Scrapy\\wiki-images-download-main\\data'
listdir = Path(dir_data)
images = list(listdir.glob("*.png"))
print("Number of images found: ", len(images))
# Iterate over the dataset and store the
# information needed
templist = []
for img_path in images:
    temp = img_path.name.split(".png")[0]
    templist.append(temp.split('_')[1])
titleplus = 'batch_' + str(int(max(templist))+1) + '_'

for runs in range(100):
    # proc = subprocess.Popen(batcmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # tmp = proc.stdout.read()
    # print(tmp)
    os.system('cmd /c "cd C:\\NeuroBrave\\Scrapy\\wiki-images-download-main\\wiki_images\\spiders & scrapy crawl paris"')
    time.sleep(30)
    alltext_to_png("data/" + titleplus + str(runs))