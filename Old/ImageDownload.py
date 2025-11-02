import requests
import json

with open('kimono_image_urls_remove.json', 'r') as f:
    image_sources = json.load(f)
imag_data = requests.get(image_sources[0]).content

startIndex = 0
endIndex = len(image_sources)

for i in range(startIndex, endIndex):
    imag_data = requests.get(image_sources[i]).content
    with open('./Kimono_Images/image_' + str(i) + '.jpg', 'wb') as handler:
        handler.write(imag_data)
    if i%100 == 0: print("Image " + str(i) + " done")
    # time.sleep(1)
    