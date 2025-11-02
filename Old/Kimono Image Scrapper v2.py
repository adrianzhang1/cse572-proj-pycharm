import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import time
import json

urlOrig = "https://search.rakuten.co.jp/search/mall/%E7%9D%80%E7%89%A9/?sid=223135"
urlPages1 = "https://search.rakuten.co.jp/search/mall/%E7%9D%80%E7%89%A9/?p="
urlPages2 = "&sid=223135"

pageStart = 2
pageEnd = 150

page1 = requests.get(urlOrig)

soup1 = BeautifulSoup(page1.content, 'html.parser')

# print(soup1.prettify())

link_wrapper = soup1.find_all('a', class_='image-link-wrapper--3XCNg')
print(link_wrapper[0])

# all_links = []
hrefs = [link['href'] for link in link_wrapper]

for i in range(pageStart,   +1):
    page = requests.get(urlPages1 + str(i) + urlPages2)

    soup1 = BeautifulSoup(page.content, 'html.parser')
    link_wrapper = soup1.find_all('a', class_='image-link-wrapper--3XCNg')
    hrefs.extend([link['href'] for link in link_wrapper])

    print(len(hrefs))
    print("Page " + str(i) + " done")
    time.sleep(1)

file_path = "kimono_hrefs.json"

with open(file_path, 'w') as f:
    json.dump(hrefs, f)

print(hrefs)
print(len(hrefs))

for w in link_wrapper:
    links = w.find_all('href')
    for link in links:
        im

image_sources = []

for w in image_wrapper:
    images = w.find_all('img')
    for img in images:
        img_url = img.get('src')
        if img_url:
            index = img_url.find('?')
            if index!=-1:
                img_url = img_url[:index]
            image_sources.append(img_url)

print(len(image_sources))
print(image_sources[1:5])

for i in range(pageStart, pageEnd+1):
    page = requests.get(urlPages1 + str(i) + urlPages2)

    soup1 = BeautifulSoup(page.content, 'html.parser')
    image_wrapper = soup1.find_all('div', class_='image-wrapper--3eWn3')
    for w in image_wrapper:
        images = w.find_all('img')
        for img in images:
            img_url = img.get('src')
            if img_url:
                index = img_url.find('?')
                if index!=-1:
                    img_url = img_url[:index]
                image_sources.append(img_url)
    print(len(image_sources))
    print("Page " + str(i) + " done")
    time.sleep(1)

print(len(image_sources))

file_path = "kimono_image_urls.json"

with open(file_path, 'w') as f:
    json.dump(image_sources, f)