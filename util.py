import math

import cv2
import numpy as np
from matplotlib import pyplot as plt


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
    ret, image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return image_bin


def detect_map(image_orig, image_bin):
    image_bin_copy = image_bin.copy()
    image_bin_copy = cv2.bitwise_not(image_bin_copy)

    h, w = image_bin.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(image_bin_copy, mask, (0, 0), 0)
    display_image(image_bin_copy)

    image_bin_copy_inv = cv2.bitwise_not(image_bin_copy)

    kernel = np.ones((2, 2), np.uint8)
    image_bin_copy_inv = cv2.dilate(image_bin_copy_inv, kernel, iterations=4)
    image_bin_copy_inv = cv2.erode(image_bin_copy_inv, kernel, iterations=3)
    display_image(image_bin_copy_inv)

    image_bin_copy_inv[np.where((image_bin_copy_inv == [0]).all(axis=1))] = [255]
    bin_to_rgb = cv2.cvtColor(image_bin_copy_inv, cv2.COLOR_GRAY2RGB)
    bin_to_rgb[np.where((bin_to_rgb == [255, 255, 255]).all(axis=2))] = [255, 0, 0]
    display_image(bin_to_rgb, False)

    cv2.imwrite("maps/detected_map.png", bin_to_rgb)

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
    A = (area(x1, y1, x2, y2, x3, y3) + area(x1, y1, x4, y4, x3, y3))

    A1 = area(x, y, x1, y1, x2, y2)

    A2 = area(x, y, x2, y2, x3, y3)

    A3 = area(x, y, x3, y3, x4, y4)

    A4 = area(x, y, x1, y1, x4, y4)

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
