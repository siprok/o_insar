"""
    Файл, содержащий вспомогательные функции 
"""


import numpy as np
from numpy import pi
from numba import njit
from itertools import product
from sys import float_info
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from cv2 import boxFilter, BORDER_REPLICATE


def axis_partition(axis_size: int, partition_num: int, intersection_num: int) -> list:
    """
    Функция формирования списка кортежей, содержащих границы полуинтервалов для разбиения матрицы по соответствующей оси
    :param axis_size: чило элементов по соответствующей оси
    :param partition_num: число интервалов разбиения
    :param intersection_num: число общих элементов для оседних интервалов
    :return: список кортежей, в котором первый элемент - индекс начало полуинтервала
                                        второй - индекс конца полуинтервала (индекс элемента, следующего за последним)

    """
    axis_slices = [(0,0)] * partition_num
    # проверка возможно ли разбить всю ось на одинаковые части и выбор соответствующих длин интервалов
    if (axis_size - intersection_num) < partition_num:
        raise ValueError("too small axis for such intersection and partition")
    rest_part = (axis_size - intersection_num) % partition_num
    if rest_part == 0:
        basic_step = (axis_size - intersection_num) // partition_num  # шаг на всех интервалах до начала пересечения, кроме последнего
        last_step = basic_step + intersection_num  # шаг последнего интервала
    else:
        basic_step = (axis_size - intersection_num) // partition_num
        last_step = axis_size - basic_step * (partition_num - 1)
    for interval in range(partition_num - 1):
        axis_slices[interval] = (basic_step * interval, basic_step * (interval + 1) + intersection_num)
    axis_slices[-1] = (-last_step, axis_size)
    return axis_slices


def dispersion_by_window(source_matrix: np.ndarray, window_step: int) -> np.ndarray:
    """
    Функция нахождения дисперсии по окнам с заданным радиусом
        :param source_matrix: исходная матрица
        :param window_step: радиус окна фильтра, не считая центральную ячейку
        :return: матрица накопитель
        """
    ksize = (2 * window_step + 1,  2 * window_step + 1)  # кортеж размеров ядра фильтра
    row_num, column_num = source_matrix.shape  # получаем размеры исходной матрицы
    # срезы матрицы, для которых значение функции определено
    rows_result_slice = slice(2 * window_step, row_num - 2 * window_step)
    columns_result_slice = slice(2 * window_step, column_num - 2 * window_step)
    accumulator = np.zeros((row_num, column_num), dtype=source_matrix.dtype)  # матрица накопителя
    mean_by_window = boxFilter(source_matrix, -1, ksize, normalize=True, borderType=BORDER_REPLICATE) # матрица мат ожиданий по окнам
    shifts_range = range(-window_step, window_step + 1)  # диапозон сдвигов по окну
    for row_shift, column_shift in product(shifts_range, shifts_range):
        rows_slice = slice(2 * window_step + row_shift, row_num - 2 * window_step + row_shift)  # срез по строкам при текущем сдвиге
        columns_slice = slice(2 * window_step + column_shift, column_num - 2 * window_step + column_shift) # срез по столбцам при текущем сдвиге
        accumulator[rows_result_slice, columns_result_slice] += np.power(source_matrix[rows_slice, columns_slice] - mean_by_window[rows_result_slice, columns_result_slice], 2)
    return accumulator / (2 * window_step + 1) ** 2

@njit
def exp_col(rows_number: int, columns_number: int, col_shift: int) -> np.ndarray:
    """
    Вспомогательная функция, возвращающая показатель степени для смещения по столбцам
    :param rows_number: число строк
    :param columns_number: число столбцов
    :param col_shift: количество столбцов на которое нужно произвести сдвиг
    :return: матрица размера (rows_number x columns_number)
    """
    return np.outer(np.ones((rows_number, 1)), np.arange(columns_number).reshape(1, -1) * col_shift / columns_number)

@njit
def exp_row(rows_number: int, columns_number: int, row_shift: int) -> np.ndarray:
    """
    Вспомогательная функция, возвращающая показатель степени для смещения по строкам
    :param rows_number: число строк
    :param columns_number: число столбцов
    :param row_shift: количество строк на которое нужно произвести сдвиг
    :return: матрица размера (rows_number x columns_number)
    """
    return np.outer(np.arange(rows_number).reshape(-1, 1), np.ones((1, columns_number)) * row_shift / rows_number)

@njit
def form_eq_rows(source_array: np.ndarray, num_columns: int) -> np.ndarray:
    """
    формирование матрицы такой, что в одной строке все значения равны, из одномерного массива
    param source_array: одномерный массив
    param num_columns: число столбцов в итоговой матрице

    return: матрица размером (source_array.size(), num_columns)
    ---------------
    a = np.arange(6)
    >>> form_eq_rows(a, 3)
    array([ [0., 0., 0.],
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
            [4., 4., 4.],
            [5., 5., 5.]])
    """
    return np.outer(source_array, np.ones(num_columns))

@njit
def form_eq_columns(source_array: np.ndarray, num_rows: int) -> np.ndarray:
    """
    формирование матрицы такой,что в одном столбце все значения равны, из одномерного массива
    param source_array: одномерный массив
    param num_rows: число строк в итоговой матрице

    return: матрица размером (num_rows, source_array.size())
    ----------------
    a = np.arange(6)
    >>> form_eq_columns(a, 3)
    array( [[0., 1., 2., 3., 4., 5.],
            [0., 1., 2., 3., 4., 5.],
            [0., 1., 2., 3., 4., 5.]])
    """
    return np.outer(np.ones(num_rows), source_array)


def find_energy_part(fft_matrix: np.ndarray, thresh_part: float) -> np.ndarray:
    """
    Функция нахождения непрерывного окна, центрированного относительно чентра матрицы,
    в котором находится thresh_part всей энергии
    (выбирается окно, значение энергии в котором ближе к thresh_part, чем в любом другом непрерывном окне с тем же соотношением сторон )
    :param fft_matrix: матрица центрированного фурье-образа
    :param thresh_part: доля энергии
    :return: массив массивов границ полуинтервалов  [[start_0, stop_0],[start_1, stop_1], ..., [start_N, stop_N]],
            где N = len(fft_matrix.shape)
    """
    shape_array = np.array(fft_matrix.shape)  # массив размерностей входной матрицы
    dims_num = shape_array.size  # число осей матрицы
    center = shape_array // 2   # массив индексов центров по соответствующим осям5
    radius = np.zeros(dims_num, dtype=int)  # массив отклонений от центра по соответствущим осям
    # посчитаем добавочное смещения по соответствующим осям для корректного указани правых границ срезов
    # сначала посчитаем из предположения четности размеров входной матрицы
    addition = np.zeros(dims_num, dtype=int)
    # изменим добавочное смещение в случае нечётности соответствующих размерностей входной матрицы
    addition[shape_array % 2 == 1] = 1
    energy_matrix = np.power(np.abs(fft_matrix), 2).astype(np.float64)  # матрица энергетического спектра
    # нормировка значений в случае угрозы переполнения
    if not float_info.max_10_exp > np.log10(np.max(energy_matrix)) + np.log10(energy_matrix.size):
        energy_matrix /= np.max(energy_matrix)
    full_energy = np.sum(energy_matrix)  # полная энергия спектра
    thresh = full_energy * thresh_part  # граничное значение энергии
    accumulator_energy = 0  # накопитель энергии в найденном окне
    border_energy = 0  # энергия на текущей границе
    # зададим длины шагов по соответствующим осям исходя из соотношения сторон входной матрицы
    step = np.ones(dims_num, dtype=int)
    # найдем наименьшим число элементов по осям матрицы
    min_ind = np.argmin(shape_array)
    min_capacity = shape_array[min_ind]
    # цикл по осям матрицы и установка соответствующих длин шагов
    for i in range(dims_num):
        step[i] = round(shape_array[i] / min_capacity)
    top_slices = [0] * dims_num  # массив срезов по верхним границам со осям
    bottom_slices = [0] * dims_num  # массив срезов по нижним границам со осям
    radius += step  # нужно для первого шага
    while all(radius < center + 1):
        border_energy = 0
        # сформируем статичные(одноэлементные) срезы по обновленному окну
        for i in range(dims_num):
            top_slices[i] = center[i] + radius[i] + addition[i] - 1
            bottom_slices[i] = center[i] - radius[i]
        # цикл суммирования по граничным значнеиям
        for i in range(dims_num):
            # делаем срез из всех элементов границы по текущей оси
            # суммирование и вычитание индекса производится для избавления от включения граничных элементов
            top_slices[i] = slice(center[i] - radius[i] + i, center[i] + radius[i] + addition[i] - i)
            bottom_slices[i] = slice(center[i] - radius[i] + i, center[i] + radius[i] + addition[i] - i)
            # непосредственно суммирование
            border_energy += np.sum(energy_matrix[tuple(top_slices)]) + np.sum(energy_matrix[tuple(bottom_slices)])
            # возврат срезов к одноэлементому варианту
            top_slices[i] = center[i] + radius[i] + addition[i] - 1
            bottom_slices[i] = center[i] - radius[i]
        # проверка превышения заданного уровня энергии
        if accumulator_energy + border_energy > thresh:
            break
        accumulator_energy += border_energy  # включаем энергию на текущей границе в накопитель
        radius += step  # расширяем границы окна
    if any(radius > center):
        mask = radius > center
        radius[mask] = center[mask]
    # проверим стои ли включать текущую границу в окно результата по близости к значению заданной границы
    elif abs(accumulator_energy + border_energy - thresh) < abs(accumulator_energy - thresh) or any(radius):
        accumulator_energy += border_energy
    else:
        radius -= step
    return np.hstack(((center - radius).reshape(-1, 1), (center + radius + addition).reshape(-1, 1)))


def min_span_transitions_matrix_cell(source_matrix: np.ndarray) -> dict:
    """
    Функция получение максимального остова переходов между ячейками матрицы
    :param source_matrix: двумерная матрица
    :return: матрица смежности в формате словаря, где ключами являются индексы в исходной матрице
    """

    # Далее вес вертикального перехода это сумма элементов (i, j) и (i + 1, j)
    #       вес горизонтального перехода это сумма элементов (i, j) и (i, j + 1)

    # Будем рассматривать матрицу надёжности как граф, 
    # вершины которого - ячейки матрицы, а рёбра - горизонтальные и вертикальные переходы

    # Найдем матрицы весов вертикальных и горизонтальных переходов переходов элементов исходной матрицы(source_matrix)
    # и матрицы соответствующих им индексов в матрице связности рассматриваемого графа
    
    # Матрица вертикальных переходов между ячейками исходной матрицы
    vertical_transition = source_matrix[:-1, :].copy()
    vertical_transition += source_matrix[1:, :]
    # Преобразуем в одномерный массив для использования в конструкторе разреженной матрицы
    vertical_transition = vertical_transition.flatten()
    # Матрица горизонтальных переходов между ячейками исходной матрицы
    horizontal_transition = source_matrix[:, :-1].copy()
    horizontal_transition += source_matrix[:, 1:]
    # Преобразуем в одномерный массив для использования в конструкторе разреженной матрицы
    horizontal_transition = horizontal_transition.flatten()

    rows_num = source_matrix.shape[0]  # число срок в исходной матрице
    columns_num = source_matrix.shape[1]  # число столбцов в исходной матрице

    # Матрица индексов по строкам в матрице вертикальных переходов
    vertical_row_indexes = np.outer(np.arange(rows_num - 1), np.ones(columns_num) * columns_num)
    vertical_row_indexes += form_eq_columns(np.arange(columns_num), rows_num - 1)
    # Преобразуем в одномерный массив для использования в конструкторе разреженной матрицы
    vertical_row_indexes = vertical_row_indexes.flatten()
    # Матрица индексов по столбцам в матрице вертикальных переходов
    vertical_col_indexes = np.outer((np.arange(rows_num - 1) + 1), np.ones(columns_num) * columns_num)
    vertical_col_indexes += form_eq_columns(np.arange(columns_num), rows_num - 1)
    # Преобразуем в одномерный массив для использования в конструкторе разреженной матрицы
    vertical_col_indexes = vertical_col_indexes.flatten()

    # Матрица индексов по строкам в матрице горизонтальных переходов
    horizontal_row_indexes = np.outer(np.arange(rows_num), np.ones(columns_num - 1) * columns_num)
    horizontal_row_indexes += form_eq_columns(np.arange(columns_num - 1), rows_num)
    # Преобразуем в одномерный массив для использования в конструкторе разреженной матрицы
    horizontal_row_indexes = horizontal_row_indexes.flatten()
    # Матрица индексов по столбцам в матрице горизонтальных переходов
    horizontal_col_indexes = np.outer(np.arange(rows_num), np.ones(columns_num - 1) * columns_num)
    horizontal_col_indexes += form_eq_columns(np.arange(columns_num - 1) + 1, rows_num)
    # Преобразуем в одномерный массив для использования в конструкторе разреженной матрицы
    horizontal_col_indexes = horizontal_col_indexes.flatten()

    # сформируем данные для инициализации матрицы связности в разреженом виде (в формате scipy.sparse.coo_matrix)
    data = np.concatenate((vertical_transition, horizontal_transition))
    del vertical_transition, horizontal_transition
    rows_indexes = np.concatenate((vertical_row_indexes, horizontal_row_indexes)).astype(int)
    del vertical_row_indexes, horizontal_row_indexes
    columns_indexes = np.concatenate((vertical_col_indexes, horizontal_col_indexes)).astype(int)
    del vertical_col_indexes, horizontal_col_indexes
    adj_side = rows_num * columns_num
    #np.clip(data, float_info.min, None, out=data)
    # инициализируем матрицу связности
    adj_matrix_mst = coo_matrix((data, (rows_indexes, columns_indexes)), shape=(adj_side, adj_side))
    del data, rows_indexes, columns_indexes

    # найдём минимальный остов
    adj_matrix_mst = minimum_spanning_tree(adj_matrix_mst.tocsr(), overwrite=True)
    adj_matrix_mst = adj_matrix_mst.todok(copy=True)
    return adj_matrix_mst


def phase_residue_mask(phase_matrix: np.ndarray) -> np.ndarray:
    """
    Функция построения логической маски матрица фазы для определения ячеек с особыми точками фазы
    :param phase_matrix: матрица фазы
    :return: матрица логической  маски, размера исходной матрицы
    """
    der_x_y = phase_matrix 
    der_x_y[:, 1:-1] = der_x_y[:, 2:] - der_x_y[:, :-2]
    der_x_y[1:-1, :] = der_x_y[2:, :] - der_x_y[:-2, :]
    der_x_y[0, :], der_x_y[-1, :] = phase_matrix[0, :], phase_matrix[-1, :]
    der_x_y[:, 0], der_x_y[:, -1] = phase_matrix[:, 0], phase_matrix[:, -1]
    der_y_x = phase_matrix 
    der_y_x[1:-1, :] = der_y_x[2:, :] - der_y_x[:-2, :]
    der_y_x[:, 1:-1] = der_y_x[:, 2:] - der_y_x[:, :-2]
    der_y_x[0, :], der_y_x[-1, :] = phase_matrix[0, :], phase_matrix[-1, :]
    der_y_x[:, 0], der_y_x[:, -1] = phase_matrix[:, 0], phase_matrix[:, -1]
    return (der_x_y != der_y_x)


def pi_mult(diff: float) -> int:
    """
    Функция, вычисляющая множитель, на который нужно домножить 2 pi, чтобы компенсировать разрыв фазы
    :param diff: разность фазы в двух ячейках матрицы
    :return : целое число
    """
    return int(0.5 * (diff  / pi + 1)) if diff > 0 else int(0.5 * (diff  / pi - 1))

