"""
    Файл, содержащий функции для совмещения РЛИ
"""


import numpy as np
import cv2 as cv
from numba import njit


'''
плохо работает на рли
def feature_based(first_source: np.ndarray, second_source: np.ndarray):
    """
    Функция определения величины смещения для наложения изображений
    Основная идея: нахождение искомых параметров смещения по особым(отличительным) точкам полутоновых изображений
    :param first_source: матрица первого полутонового изображения (двумерная матрица)
    :param second_source: матрица второго полутонового изображения (двумерная матрица))
    :return: кортеж(rows_correction, columns_correction),
                где rows_correction - число строк, на которое нужно сместить второе изображение вниз;
                    columns_correction - число столбцов, на которое нужно сместить второе изображение вправо
    """


    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15


    # Detect ORB features and compute descriptors.
    orb = cv.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(first_source, None)
    keypoints2, descriptors2 = orb.detectAndCompute(second_source, None)

    # Match features.
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv.drawMatches(first_source, keypoints1, second_source, keypoints2, matches, None)
    cv.imwrite("matches.jpg", imMatches)
    """
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv.findHomography(points1, points2, cv.RANSAC)

    # Use homography
    height, width, channels = second_source.shape
    im1Reg = cv.warpPerspective(first_source, h, (width, height))

    return im1Reg, h
    """
'''