"""
    Скрипт для сохранения отображения значений матрицы в виде полутонового изображения
    Аргументы:
        source_path [str] - путь к исходному файлу
        --destination_path [str]- путь к файлу,в который будет сохранен результат 
        --algorithm [str] - способ преобразования диапозона значений       
        --azimuth_start [int] - индекс начала участка обработки по строкам
        --azimuth_width [int] - ширина участка обработки по строкам (в количестве дискретов)
        --range_start [int] - индекс начала участка обработки по столбцам
        --range_width [int] - ширина участка обработки по столбцам (в количестве дискретов)
    Пример использования:
    python ./npy_trfm_bmp.py ./processing_sources/amplitude.npy --destination_path ./processing_results/amplitude.bmp --algorithm linear --azimuth_start 120 --azimuth_width 500 --range_start 360 --range_width 700                                 
"""


import numpy as np
from sys import float_info
import argparse
from numba import njit
import os
import cv2 as cv


@njit(cache=True)
def histeq(image: np.ndarray, nbr_bins=256):
    """Выравнивание гистограммы полутонового изображения"""

    # получить гистограмму изображения
    imhist, bins = np.histogram(image.flatten(), nbr_bins)
    cdf = imhist.cumsum()  # функция распределения
    cdf = 255 * cdf / cdf[-1]  # нормирование

    # используем линейную интерполяцию cdf для нахождения значений новых пикселей
    im2 = np.interp(image.flatten(), bins[:-1], cdf)
    return im2.reshape(image.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converting array data from file to bmp, using histogram equalizing.')
    parser.add_argument('source_path', type=str, help='Path to file containing source array')
    parser.add_argument('--destination_path', type=str, help='Path to save image file')
    parser.add_argument('--algorithm', type=str, help='part_lin, histeq, linear')
    parser.add_argument('--azimuth_start', type=int)
    parser.add_argument('--azimuth_width', type=int)
    parser.add_argument('--range_start', type=int)
    parser.add_argument('--range_width', type=int)

    args = parser.parse_args()
    algorithm = args.algorithm or "histeq"
    array = np.load(args.source_path)

    rows_start = args.azimuth_start or 0
    rows_stop = args.azimuth_width or array.shape[0]
    rows_stop += rows_start

    columns_start = args.range_start or 0
    columns_stop = args.range_width or array.shape[1]
    columns_stop += columns_start

    array = array[rows_start:rows_stop, columns_start:columns_stop]

    destination = args.destination_path or os.path.splitext(args.source_path)[0] + "_" + algorithm + ".bmp"

    if np.min(array) < 0:
        array -= np.min(array)

    if algorithm == 'linear':
       array = (array - np.min(array)) * 255 / (np.max(array) - np.min(array))
    elif algorithm == 'part_lin':
        mean = np.mean(array)
        low_threshold = np.percentile(array, 25)
        top_threshold = np.percentile(array, 95)
        np.clip(array, low_threshold, top_threshold, array)
        array = np.where(array < mean, array * 0.5, array * 5)
        array = 255 * (array - np.min(array)) / (np.max(array) - np.min(array))
    elif algorithm == 'histeq':
        low_threshold = np.percentile(array, 37)
        top_threshold = np.percentile(array, 87)
        median = np.median(array)
        np.clip(array, low_threshold, top_threshold, array)
        np.clip(array, float_info.min, None, array)
        if median != 0 and median != 1:
            array = histeq(np.log(array) / np.log(median))
        else:
            array = histeq(np.log(array) / np.log(median + float_info.min))
    cv.imwrite(destination, array.astype('uint8'))