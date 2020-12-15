#!/usr/bin/env python
# Li Xue
#  3-Jul-2020 16:01
'''
Resize images under a folder to target pixel size.
Usage:
    python resize_img.py <folder> <target_row_pixel> <target_col_pixel>

Example:
    python resize_img.py ~/raw_images/ 150 150
'''

import sys
import os
import re
from os import listdir
import shutil
import glob
import imageio
from PIL import Image
from tqdm import tqdm

def resize_imgs(img_dir, row_pixel, col_pixel):

    # resize images
    imgFLs = [f for f in glob.glob(f"{img_dir}/*") if re.search('(jpg|png|jpeg)',f, flags=re.IGNORECASE) ]
    print(f"There are {len(imgFLs)} images under {img_dir}")

    for imgFL in tqdm(imgFLs):
        img = Image.open(imgFL)
        img_new = img.resize((row_pixel, col_pixel))

        newFL = os.path.splitext(imgFL)[0] + '_resized.jpg'
        try:
            img_new.save(newFL)
        except:
            print(f"Error: cannot save {imgFL}")

    print(f"Resized images: {img_dir}/xx_resized.jpg")

def check_input(args):
    if len(args) !=3:
        sys.stderr.write(__doc__)
        sys.exit(0)
    img_dir = args[0]
    row_pixel = float(args[1])
    col_pixel = float(args[2])

    if not os.path.exists(img_dir):
        sys.stderr.write(f"{img_dir} does not exist")
        sys.exit(1)

    if not ( row_pixel.is_integer() and col_pixel.is_integer() ):
        sys.stderr.write(f"target_row_pixel and target_col_pixel have to be integer")
    return img_dir, row_pixel, col_pixel

def main():
    img_dir, row_pixel, col_pixel = check_input(sys.argv[1:])
    resize_imgs(img_dir, row_pixel, col_pixel)

if __name__ == '__main__':
    main()
