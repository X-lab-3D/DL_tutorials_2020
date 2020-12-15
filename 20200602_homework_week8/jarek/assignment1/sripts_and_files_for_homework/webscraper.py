import urllib.request
import re
import os

__author__ = 'Daniel-Tobias Rademaker', 'Jarek Miron van Dijk'


website = 'https://emojipedia.org/apple/'
img_locations = []

def getHTML(website): # Retrieve the source code from any website
    req = urllib.request.Request(website)
    req.add_header('User-Agent', '')
    response = urllib.request.urlopen(req)
    html = response.read() # Read the source
    return html


html = str(getHTML(website)) # get the source code
print(html)
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
