import numpy as np
import pandas as pd
from collections import Counter
from itertools import chain
import random
from openai import OpenAI

df = pd.read_csv('Labeled Color Connections - Sheet1.csv')

numberOfColors = input('Enter number of colors wanted: ')
numberOfColors = int(numberOfColors)

requestedColors = []

for i in range(numberOfColors):
    if(i < numberOfColors - 1):
        requestedColors.append(input('Enter a requested Hex value that is a multiple of 16 (press enter to skip): '))

requestedColors = [x for x in requestedColors if x != '']
potentialColors = []
selectedColors = 0

for i in range(np.size(requestedColors)):
        row = df.loc[df['Colors'] == requestedColors[i]].apply(pd.to_numeric, errors='coerce').squeeze()
        potentialColors.append(row.drop('Colors').nlargest(50).index)
        selectedColors += 1

sortedColors = []
test = []

for i in range(selectedColors):
    sortedColors.append(potentialColors[i].tolist())
    test = list(chain.from_iterable(sortedColors))

counter = Counter(test)
duplicates = [item for item, count in counter.items() if count > 1]

finalColors = []

finalColors.append(requestedColors)

for i in range(numberOfColors - np.size(requestedColors)):
    if(selectedColors > 1):
        finalColors.append(random.choice(duplicates))
    else:
        finalColors.append(random.choice(test))
print(finalColors)

client = OpenAI(api_key='replace')

prompt = f"""
Evaluate the color palette for a kimono:
{finalColors}
Give a 1-10 aesthetic rating and notes on the colory harmony
"""

response = client.chat.completions.create(model="gpt-4.1", messages=[{"role": "user", "content": prompt}])

print(response.choices[0].message.content)