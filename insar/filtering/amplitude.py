"""
    Файл, содержащий функции для фильтрации амплитудной составляющей сигнала
"""


import numpy as np
from cv2 import boxFilter, BORDER_REPLICATE


def kuan_argument_checker(input_matrix: np.ndarray, window_step: int, enl=1, rli_type='amplitude') -> None:
    """
    Проверка входных данных фильтра Куана на допустимые значения
    input_matrix - амплитудная матрица РЛИ
    window_step  - радиус окна фильтра
    enl  - эквивалентное число некогерентных накоплений
    rli_type - тип РЛИ ('amplitude', 'power')
    """
    if rli_type not in {'amplitude', 'power'}:
        raise ValueError('rli_type')

    if not isinstance(window_step, int):
        raise TypeError('window_step')
    if window_step < 1:
        raise ValueError('window_step')

    if not isinstance(enl, int):
        raise TypeError('enl', int, enl)
    if enl < 1:
        raise ValueError('enl')

    if not isinstance(input_matrix, np.ndarray):
        raise TypeError('input_matrix')
    if input_matrix.shape[0] < 2 * window_step + 1 or input_matrix.shape[0] < 2 * window_step + 1:
        raise ValueError('input_matrix too small')


def kuan_filter(input_matrix: np.ndarray, window_step: int, enl=1, rli_type='amplitude') -> np.ndarray:
    """
    Алгоритм Куана фильтрации спекл-шума
    input_matrix - матрица РЛИ
    window_step  - радиус окна фильтра
    enl  - эквивалентное число некогерентных накоплений
    rli_type - тип РЛИ ('amplitude', 'power')

    """

    rvmn = 0.273 / enl if rli_type == 'amplitude' else 1 / enl  # относительная дисперсия мультипликативного шума
    dvdr = 1 + rvmn  # делитель для нахождения весового коэффициента

    ksize = (2 * window_step + 1,  2 * window_step + 1)  # кортеж размеров ядра фильтра

    # матрица ячейке (i, j) которой соотвествует мат.ожидание окна с центром в точке (i + window_step, j + window_step)
    mean_matrix = boxFilter(input_matrix, -1, ksize, normalize=True, borderType=BORDER_REPLICATE)

    # матрица ячейке (i, j) которой соотвествует мат.ожидание квадратов окна
    # с центром в точке (i + window_step, j + window_step)
    mean_sqr_matrix = boxFilter(input_matrix ** 2, -1, ksize, normalize=True, borderType=BORDER_REPLICATE)

    # матрица весовых коэффициентов
    k = (1 - rvmn * mean_matrix ** 2 / (mean_sqr_matrix - mean_matrix ** 2))
    # аналог np.clip через функцию, поддерживаемую библиотекой numba
    k = np.where(k < 0, 0, k)
    k /= dvdr
    result = input_matrix.copy()
    result = mean_matrix + k * (input_matrix - mean_matrix)
    return result
