"""
    Скрипт для развертки фазы и сохранения результата в виде npy файла
    Аргументы:
        source_path [str] - путь к исходному файлу
        --destination_path [str]- путь к файлу,в который будет сохранен результат 
        --algorithm [str] - способ развертки фазы
        --quality_map [str] - метод построения матрицы качества, необходимой лоя развертки
        --window_step [int] - радиус окна, используемого при построении матрицы качества
        --azimuth_start [int] - индекс начала участка обработки по строкам
        --azimuth_width [int] - ширина участка обработки по строкам (в количестве дискретов)
        --range_start [int] - индекс начала участка обработки по столбцам
        --range_width [int] - ширина участка обработки по столбцам (в количестве дискретов)
    Пример использования:
    python ./npy_unwrap_npy.py ./processing_sources/phase.npy --destination_path ./processing_results/unwrapped_phase.npy --algorithm relnp --quality_map sd --azimuth_start 120 --azimuth_width 500 --range_start 360 --range_width 700                                 
"""


import argparse
import numpy as np
from time import time
import os
from insar.unwrapping.algorithms import relnp, itohs_unwrapping, td_itohs_unwrapping
from insar.unwrapping import quality_maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unwrapping phase matrix')
    parser.add_argument('source_path', type=str, help='Path to npy file with wrapped phase')
    parser.add_argument('--destination_path', type=str, help='Path to save unwrapped phase')
    parser.add_argument('--algorithm', help='relnp, itohs, td_itohs', type=str)
    parser.add_argument('--quality_map', help='sd (default; mean second_differ);\n'+
                                              'cm (mean correlation_map);\n'+
                                              'pcm (mean pseudo_correlation_map)', type=str)
    parser.add_argument('--window_step', type=int)
    parser.add_argument('--azimuth_start', type=int)
    parser.add_argument('--azimuth_width', type=int)
    parser.add_argument('--range_start', type=int)
    parser.add_argument('--range_width', type=int)
    args = parser.parse_args()
    alg = args.algorithm or "relnp"
    qm = args.quality_map or "sd"
    destination = args.destination_path or os.path.splitext(args.source_path)[0] + "_unwrapped" + "_" + alg + "_" + qm +".npy"
    phase_matrix = np.load(args.source_path)

    rows_start = args.azimuth_start or 0
    if rows_start < 0:
        rows_start = 0
    rows_stop = args.azimuth_width or phase_matrix.shape[0] - rows_start
    rows_stop += rows_start

    if rows_stop > phase_matrix.shape[0]:
        rows_stop = phase_matrix.shape[0]

    columns_start = args.range_start or 0
    if columns_start < 0:
        columns_start = 0
    columns_stop = args.range_width or phase_matrix.shape[1]
    columns_stop += columns_start

    if columns_stop > phase_matrix.shape[1]:
        columns_stop = phase_matrix.shape[1]

    window_step = args.window_step or 1

    wrapped_phase = phase_matrix[rows_start:rows_stop, columns_start:columns_stop]

    if not args.algorithm or args.algorithm == 'relnp':
        if not args.quality_map or args.quality_map == "sd":
            relative_matrix = quality_maps.phase_second_diff(wrapped_phase)
        elif args.quality_map == "pcm":
            relative_matrix = quality_maps.pseudo_correlation(wrapped_phase, window_step)
        elif args.quality_map == "pdv":
            relative_matrix = quality_maps.phase_derivative_variance(wrapped_phase, window_step)
        elif args.quality_map == "mpg":
            relative_matrix = quality_maps.phase_max_grad(wrapped_phase, window_step)
        else:
            raise ValueError("Wrong quality_map")
        start = time()
        s = relnp(wrapped_phase, relative_matrix)
        end = time()
    else:
        if args.algorithm == 'itohs':
            unwrap_func = itohs_unwrapping
        elif args.algorithm == 'td_itohs':
            unwrap_func = td_itohs_unwrapping
        else:
            raise ValueError('Wrong algorithm label')
        start = time()
        s = unwrap_func(wrapped_phase)
        end = time()

    print('It need {} seconds'.format(end - start))
    np.save(destination, s)
