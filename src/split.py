#!/usr/bin/env python3

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import seaborn as sns
from collections import deque
import operator

BLOCK_SIZE = 9
C = 0

def binarize_cv():
    in_name = '0'
    image_raw = cv2.imread('%s.jpg' % in_name, cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image_raw, cv2.COLOR_RGB2GRAY)
    image_bin_mean = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, C)
    image_bin_gaussian = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, C)

    titles = [in_name + '_Original', in_name + '_Grayscale', in_name + '_Mean', in_name + '_Gaussian']
    images = [image_raw, image_gray, image_bin_mean, image_bin_gaussian]

    for i in range(4):
    #    plt.subplot(2, 2, i+1)
    #    plt.imshow(images[i])
    #    plt.title(titles[i])
    #    plt.xticks([]), plt.yticks([])
        cv2.imwrite(titles[i] + '.jpg', images[i])
    #plt.show()

def binarize(img, threshold):
    gray_img = Image.new('L', img.size, 0)
    w, h = img.size
    gray_data = gray_img.load()
    img_data = img.load()
    for x in range(w):
        for y in range(h):
            gray_data[x, y] = 255 * (img_data[x, y][0] > threshold or
                                     img_data[x, y][1] > threshold or
                                     img_data[x, y][2] > threshold)
    return gray_img

def remove_connected_black(gray_img):
    X, Y = gray_img.size
    gray_data = gray_img.load()
    span_img = Image.new('L', gray_img.size, 255)
    span_data = span_img.load()
    depth = [80, 160]
    index = 0
    stats = []

    def span(x, y, index):
        stats.append(0)
        q = deque()
        q.append((x,y))
        while len(q) != 0:
            x, y = q.popleft()
            span_data[x, y] = index
            stats[index] += 1
            for (xx, yy) in [(x-1,y-1),(x-1,y),(x-1,y+1),(x,y-1),(x,y+1),(x+1,y-1),(x+1,y),(x+1,y+1)]:
                if xx >= 0 and xx < X and yy >= 0 and yy < Y\
                    and gray_data[xx, yy] != 255\
                    and span_data[xx, yy] == 255\
                    and (xx, yy) not in q:
                    q.append((xx, yy))
#            print([(x,y) for (x,y) in q])
        
    for x in range(X):
        for y in range(Y):
            # skip white or spanned pixels
            if gray_data[x, y] != 255 and span_data[x, y] == 255:
                span(x, y, index)
                index += 1
    stats_sorted = sorted(zip(range(len(stats)), stats), key=operator.itemgetter(1), reverse=True)
    top1_key = stats_sorted[0][0]
    print(stats_sorted)
    for x in range(X):
        for y in range(Y):
            if span_data[x, y] == top1_key:
                gray_data[x, y] = 255

#    fig, ax = plt.subplots(1,1)
#    sns.barplot(list(range(len(stats))), sorted(stats), palette='Set3', ax=ax)
    return span_img

def main1():
    img = Image.open('0.jpg')
    gray_img = binarize(img, 60)
    span_img = remove_connected_black(gray_img)
    gray_img.show()
    span_img.show()
#    plt.show()

def main2():
    binarize_cv()

if __name__ == '__main__':
    main2()