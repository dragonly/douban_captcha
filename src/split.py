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
    # image_gray = cv2.dilate(image_gray, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
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
    #    plt.show()

def binarize(image_color, threshold):
    X, Y = image_color.shape[:2]
    image_bin = np.zeros(image_color.shape)
    image_dilated = cv2.dilate(image_color, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    for x in range(X):
        for y in range(Y):
            image_bin[x, y] = 255 * (image_dilated[x, y, 0] > threshold or
                                     image_dilated[x, y, 1] > threshold or
                                     image_dilated[x, y, 2] > threshold)
    return image_bin

def remove_connected_black(image_bin):
    X, Y = image_bin.shape
    gray_data = gray_img.load()
    image_span = np.zeros()
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
    image_color = cv2.imread('0.jpg', cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
    image_bin = binarize(image_raw, 60)
    # image_span = remove_connected_black(image_bin)
    cv2.imwrite('gray.jpg', image_gray)
    cv2.imwrite('bin.jpg', image_bin)
    # cv2.imwrite('span.jpg', image_span)
#    plt.show()

def main2():
    binarize_cv()

if __name__ == '__main__':
    main1()
