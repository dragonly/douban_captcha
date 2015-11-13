#!/usr/bin/env python3

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import seaborn as sns
from collections import deque
import operator
import glob

def binarize_cv(image_gray):
    BLOCK_SIZE = 29
    C = 20
    image_gray = cv2.dilate(image_gray, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    image_bin_mean = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, C)
    image_bin_gaussian = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, C)
    cv2.imwrite('bin_mean.jpg', image_bin_mean)
    cv2.imwrite('bin_gaussian.jpg', image_bin_gaussian)

def binarize(image_color, threshold):
    X, Y = image_color.shape[:2]
    image_bin = np.zeros(image_color.shape[:2], dtype=np.uint8)
    for x in range(X):
        for y in range(Y):
            image_bin[x, y] = 255 * (image_color[x, y, 0] > threshold or
                                     image_color[x, y, 1] > threshold or
                                     image_color[x, y, 2] > threshold)
    return image_bin

def cutout(image_bin):
    X, Y = image_bin.shape
    image_span = np.ones(image_bin.shape) * 255
    image_bin_span = image_bin.copy()
    index = 0
    stats = []

    def span(x, y, index):
        stats.append(0)
        q = deque()
        q.append((x,y))
        while len(q) != 0:
            x, y = q.popleft()
            image_span[x, y] = index
            stats[index] += 1
            for (xx, yy) in [(x-1,y),(x,y-1),(x+1,y),(x,y+1),(x-1,y-1),(x-1,y+1),(x+1,y-1),(x+1,y+1)]:
                if xx >= 0 and xx < X and yy >= 0 and yy < Y\
                    and image_bin[xx, yy] != 255\
                    and image_span[xx, yy] == 255\
                    and (xx, yy) not in q:
                    q.append((xx, yy))
#            print([(x,y) for (x,y) in q])

    for x in range(X):
        for y in range(Y):
            # skip white or spanned pixels
            if image_bin[x, y] != 255 and image_span[x, y] == 255:
                span(x, y, index)
                index += 1
    stats_sorted = sorted(zip(range(len(stats)), stats), key=operator.itemgetter(1), reverse=True)
    print(stats_sorted)
    top_keys = set()
    l = len(stats_sorted)
    for i in stats_sorted:
    # small threshold kills the innocents as well :(
        if i[1] > 400:
            top_keys.add(i[0])
    
    for x in range(X):
        for y in range(Y):
            if image_span[x, y] in top_keys:
                image_bin_span[x, y] = 255

#    fig, ax = plt.subplots(1,1)
#    sns.barplot(list(range(len(stats))), sorted(stats), palette='Set3', ax=ax)
    return image_bin_span
    

def process(filename):
    image_color = cv2.imread(filename, cv2.IMREAD_COLOR)
    image_median = cv2.medianBlur(image_color, 3)
    image_bin = binarize(image_median, 21)
#    image_bin_tophat = cv2.morphologyEx(image_bin, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
    image_cut = cutout(image_bin)
    image_erode = cv2.erode(image_cut, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)), iterations=1)

    cv2.imwrite('%s_median.png' % filename, image_median)
    cv2.imwrite('%s_bin.png' % filename, image_bin)
    cv2.imwrite('%s_cut.png' % filename, image_cut)
    cv2.imwrite('%s_erode.png' % filename, image_erode)

if __name__ == '__main__':
    morph_size = 9
    BLOCK_SIZE = 5
    C = 2
 
#    process('0.jpg')   
    for filename in glob.glob('./data/*.jpg'):
#    for filename in glob.glob('0.jpg'):
        print(filename)
        process(filename)
