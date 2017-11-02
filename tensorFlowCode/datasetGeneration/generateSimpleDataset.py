"""A script to generate a Thai letter dataset

This script generate 64x64 pictures of all Thai letters in different fonts and
colors and puts them in a subfolder of its directory. The fonts used are
provided by:

https://github.com/lannainnovation/thai-font-collection

To use this script download the zip from GitHub and extract it in the parent
directory of the script.
"""

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
import shutil

# Letters of the Thai alphabet
thaiAlphabet = ['ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ',
                'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ',
                'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ฤ',
                'ล', 'ฦ', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ', 'ฯ', 'ะ',
                ' ั', 'า', 'ำ', '\u0e34', '\u0e35', '\u0e36', '\u0e37',
                '\u0e38', '\u0e39', '\u0e3a', 'เ', 'แ', 'โ', 'ใ', 'ไ', 'ๅ',
                'ๆ', '\u0e47', '\u0e48', '\u0e49', '\u0e4a', '\u0e4b',
                '\u0e4c', '\u0e4d', '\u0e4e', '๏', '๐', '๑', '๒', '๓',
                '๔', '๕', '๖', '๗', '๘', '๙', '๚', '๛']

# Get font paths from font directory
fontDir = './thai-font-collection-master/downloadable-free-thai-fonts/'
fontNames = []
for (dirPath, dirNames, fileNames) in os.walk(fontDir):
    for fileName in fileNames:
        if fileName[-4:] == '.ttf':
            fontNames.append(os.path.join(dirPath, fileName))

# Text colors
colors = [(0, 0, 0),
          (255, 0, 0),
          (0, 255, 0),
          (0, 0, 255),
          (255, 255, 0),
          (0, 255, 255)]

print('Generating ' + repr(len(fontNames)*len(colors)) +
      ' examples of each letter...')

# Initialize empty folder for images
if os.path.exists('./images/'):
    shutil.rmtree('./images/')
os.makedirs('./images/')

# For each letter in each font and color create one 64x64 image
for i, letter in enumerate(thaiAlphabet):
    for f, fontName in enumerate(fontNames):
        for c, color in enumerate(colors):
            img = Image.new('RGB', (64, 64), "white")
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(fontName, 35)
            draw.text((25, 10), letter, fill=color, font=font)
            img.save('./images/' +
                     'Image' + repr(i) +
                     '_' + repr(f) +
                     '_' + repr(c) + '.png')
