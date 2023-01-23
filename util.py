import os
import cv2
import matplotlib
import math
from matplotlib import pyplot as plt
import numpy as np
from constants import *

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.show()


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return image_bin


def select_roi(image_orig, image_bin):
    # display_image(image_bin)
    im_floodfill = image_bin.copy()
    im_floodfill = cv2.bitwise_not(im_floodfill)

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = image_bin.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 0)
    display_image(im_floodfill)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    kernel = np.ones((2, 2), np.uint8)
    im_floodfill_inv = cv2.dilate(im_floodfill_inv, kernel, iterations=4)
    im_floodfill_inv = cv2.erode(im_floodfill_inv, kernel, iterations=3)
    display_image(im_floodfill_inv)

    # Combine the two images to get the foreground.
    im_out = im_floodfill | im_floodfill_inv
    # display_image(backtorgb)
    im_floodfill_inv[np.where((im_floodfill_inv == [0]).all(axis=1))] = [255]
    backtorgb = cv2.cvtColor(im_floodfill_inv, cv2.COLOR_GRAY2RGB)
    backtorgb[np.where((backtorgb == [255, 255, 255]).all(axis=2))] = [255, 0, 0]
    display_image(backtorgb, False)

    cv2.imwrite("maps/detected_map.png", backtorgb)

    return image_orig


def get_corners(center, angle, length):
    left_top = [center[0] + math.cos(math.radians(360 - (angle + 30))) * length,
                center[1] + math.sin(math.radians(360 - (angle + 30))) * length]
    right_top = [center[0] + math.cos(math.radians(360 - (angle + 150))) * length,
                 center[1] + math.sin(math.radians(360 - (angle + 150))) * length]
    left_bottom = [center[0] + math.cos(math.radians(360 - (angle + 210))) * length,
                   center[1] + math.sin(math.radians(360 - (angle + 210))) * length]
    right_bottom = [center[0] + math.cos(math.radians(360 - (angle + 330))) * length,
                    center[1] + math.sin(math.radians(360 - (angle + 330))) * length]

    # right_top, left_top, left_bottom, right_bottom
    return [left_top, right_top, left_bottom, right_bottom]


def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) +
                x2 * (y3 - y1) +
                x3 * (y1 - y2)) / 2.0)


def check(x1, y1, x2, y2, x3, y3, x4, y4, x, y):
    # Calculate area of rectangle ABCD
    A = (area(x1, y1, x2, y2, x3, y3) + area(x1, y1, x4, y4, x3, y3))

    # Calculate area of triangle PAB
    A1 = area(x, y, x1, y1, x2, y2)

    # Calculate area of triangle PBC
    A2 = area(x, y, x2, y2, x3, y3)

    # Calculate area of triangle PCD
    A3 = area(x, y, x3, y3, x4, y4)

    # Calculate area of triangle PAD
    A4 = area(x, y, x1, y1, x4, y4)

    # Check if sum of A1, A2, A3
    # and A4 is same as A
    suma = A1 + A2 + A3 + A4
    return A == suma


def check_if_center_is_in_corners(x1, y1, x2, y2, x3, y3, x4, y4, x, y):
    max_x = max(x1, x2, x3, x4)
    min_x = min(x1, x2, x3, x4)

    max_y = max(y1, y2, y3, y4)
    min_y = min(y1, y2, y3, y4)

    return min_x <= x <= max_x and min_y <= y <= max_y


def detect_another_car(image, my_corners):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_gray = image_gray(image)
    th, bin_img = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)
    bin_img = cv2.dilate(bin_img, kernel, iterations=4)

    bin_img = cv2.erode(bin_img, kernel, iterations=10)

    # bin_img = cv2.bitwise_not(bin_img)
    #
    # dist_transform = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)  # DIST_L2 - Euklidsko rastojanje
    # ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    # display_image(bin_img)

    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        if w > 35 and h > 25 and w < 140 and h < 140:
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # display_image(image)
            # cv2.rectangle(image, (int(my_corners[2][0]), int(my_corners[2][1])), (int(my_corners[0][0]), int(my_corners[0][1])), (0, 255, 0), 2)
            # cv2.circle(image, (int(my_corners[0][0]), int(my_corners[0][1])), 5, (255, 255, 255), 2) #right top
            # cv2.circle(image, (int(my_corners[1][0]), int(my_corners[1][1])), 5, (0, 0, 0), 2) #left top
            # cv2.circle(image, (int(my_corners[2][0]), int(my_corners[2][1])), 5, (255, 0, 255), 2) #left bottom
            # cv2.circle(image, (int(my_corners[3][0]), int(my_corners[3][1])), 5, (0, 255, 255), 2)# right bottom
            # cv2.circle(image, (int(x + w // 2), int(y + h // 2)), 5, (50, 50, 50), 2)
            if not check_if_center_is_in_corners(my_corners[0][0], my_corners[0][1],
                                                 my_corners[1][0], my_corners[1][1],
                                                 my_corners[2][0], my_corners[2][1],
                                                 my_corners[3][0], my_corners[3][1],
                                                 x + w // 2, y + h // 2):
                # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # display_image(image)
                return x, y, w, h
    return None

def get_constants_main(map_index):
    f = open("maps/starting_positions")
    for i, line in enumerate(f.readlines()):
        if i == map_index:
            tokens = line.split("|")
            MAP_NAME = tokens[0]

    MAP = "maps/" + MAP_NAME + ".png"

    PICKLE_FOLDER = "results/" + MAP_NAME + "_map_nice_showcase/"

    return MAP, PICKLE_FOLDER

def get_constants_car(map_index):
    f = open("maps/starting_positions")
    for i, line in enumerate(f.readlines()):
        if i == map_index:
            tokens = line.split("|")
            BLUE_POSITION = [float(tokens[1].split(",")[0]), float(tokens[1].split(",")[1])]
            RED_POSITION = [float(tokens[2].split(",")[0]), float(tokens[2].split(",")[1].strip())]

    return BLUE_POSITION, RED_POSITION