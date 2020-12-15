#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:16:22 2020

@author: farzanehmeimandi
"""

from urllib.request import urlopen
from urllib.request import Request
import re
import os



website = 'https://emojipedia.org/apple/'
img_locations = []

def getHTML(website): # Retrieve the source code from any website
    req = Request(website)
    req.add_header('User-Agent', '')
    response = urlopen(req)
   # html = response.read() # Read the source
    return response

html = getHTML(website) # get the source code
refs = re.findall('<a href="/(.*?)/">\n', html)# Retrieve all emoji classes
refs = refs[:92] # only interested in 'face' emojis
refs.remove('money-mouth-face') # Remove too strange faces
refs.remove('upside-down-face') # Remove too strange faces
refs.remove('rolling-on-the-floor-laughing')# Remove too strange faces
refs.remove('exploding-head')# Remove too strange faces
refs.remove('face-screaming-in-fear')# Remove too strange faces

base_ref = 'https://emojipedia.org/%s/' # The base site all faces have in common
for emoji in refs:
    print(emoji)
    html = getHTML(base_ref % emoji) # get html per emoji class
    img_locations += re.findall('<img src="(.*?).png"', html) # extract all image locations

# Create a directory to save the images in
try: os.mkdir('imgs')
except:pass

# Donwload the actual png images
for i, n in enumerate(img_locations):
    if 'docomo' not in n and 'au-kddi' not in n:
        name = '%s_%i.png' % (n.split('/')[-1], i)
        os.system('cd imgs; wget -q --output-document=%s %s.png' % (name, n))