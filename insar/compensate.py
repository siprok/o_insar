"""
    Файл, содержащий функции компенсации фазовых набегов
"""


import numpy as np
from scipy.signal.windows import get_window
from scipy.interpolate import CubicSpline
from scipy.stats import entropy
from insar.auxiliary import exp_col, exp_row


def grid_shift_interpolate(source_matrix: np.ndarray, az_stripe_width=30, r_stripe_width=30, eq_part=1, max_exp=5, starting_axis=1, td_flag=False):
    """
        Функция компенсации фазовых набегов по строкам и столбцам путем сдвига Фурье-спектра.
        !!!ВНИМАНИЕ!!! ВХОДНАЯ МАТРИЦА ДОЛЖНЫ ИМЕТЬ ЧЁТНЫЕ РАЗМЕРЫ, В ПРОТИВНОМ СЛУЧАЕ КРАЙНИЕ СТОЛБЕЦ И СТРОКА НЕ БУДУТ ОБРАБАТЫВАТЬСЯ
        Матрица разбивается на квадраты размером (az_stripe_width + n_az_eq_layers) x (r_stripe_width + n_r_eq_layers),
    где n_az_eq_layers и n_r_eq_layers число уровней по которым производится согласование
    :param source_matrix: комплекснозначная матрица интерферограммы
    :param az_stripe_width: сторона окна анализа по азимуту (количество столбцов)
    :param r_stripe_width: сторона окна анализа по дальности (количество строк)
    :param eq_part: доля от ширины полосы по соответствующей оси которая будет доавлена с каждой стороны квадрата для выравнивания относительно других частей
    :param max_exp: максимальна степень для изменения размеров сетки интерполяции при поиске величины смещения спектра
    :param td_flag: флаг осуществления смещения спектра по двум осям по очереди
    :param starting_axis: ось вдоль которой будет производится сдвиг в первую очередь.
                         В случае td_flags=False сдвиг производится только по этой оси;
                         в случае td_flags=True сначала производится сдвиг во всем окне ([az_stripe_width * eq_part] x [r_stripe_width * eq_part]) по указанной оси, 
                         потом во всем окне по другой оси
    :return: матрица с компенсированными эффектами искажения фазы
    """
    source_shape = np.array(source_matrix.shape)  # размеры исходной матрицы
    # Проверка на чётность размеров матрицы
    for ax in range(source_shape.size):
        if source_shape[ax] % 2:
            source_shape[ax] -= 1
    conj_prod = source_matrix[:source_shape[0],:source_shape[1]]
    if az_stripe_width > conj_prod.shape[0]:
        az_stripe_width = conj_prod.shape[0]
    if r_stripe_width > conj_prod.shape[1]:
        r_stripe_width = conj_prod.shape[1]
    if td_flag:
        contr_starting_axis = (starting_axis + 1) % 2  # индекс оси, вдоль которой будет происходить сдвиг во вторую очередь
        shifting_order = [starting_axis, contr_starting_axis]  # упорядоченный список идексов осей для осуществления сдвига спектра 
    else:
        shifting_order = [starting_axis]  # упорядоченный список идексов осей для осуществления сдвига спектра

    square_width = np.array([az_stripe_width, r_stripe_width])  # размеры окна анализа без учета слоев пересеения для согласования
    num_eq_layers = (square_width * eq_part).astype(int)  # число слоев персечения между предыдущи и следующим окном для согласования
    num_eq_layers = np.where((square_width + num_eq_layers) % 2 == 1, num_eq_layers + 1, num_eq_layers)

    result = np.zeros(conj_prod.shape, dtype=conj_prod.dtype)

    square_slices = [slice(0)] * 2  # массив срезов по осям исходной матрицы для выделения анализируемого окна
    shifting_slices = [slice(0)] * 2  # массив срезов для индексации осей окна при усреднении фурье-образа
    connection_slices = [slice(0)] * 2 # массив срезов по анализируемому окну для согласования его с результирующей матрицей
    result_slices = [slice(0)] * 2  # массив срезов по осям результирующей матрицы для согласования и включения анализируемого окна

    square_shape = square_width + num_eq_layers  # число строк и столбцов в окне анализа
    for i, shape in enumerate(source_shape):
        if square_shape[i] > shape:
            square_shape[i] = shape
    center_indexes = square_shape // 2 - 1 # индексы центрального элемента в окне анализа

    freq_mult = 2 ** (max_exp - 1)

    less_border = np.array([0, 0])  # меньшие границы по соответствующим осям
    more_border = square_width + num_eq_layers # бОльшие границы по соответствующим осям

    while True:
        for ax in range(2):
            square_slices[ax] = slice(less_border[ax], more_border[ax])  # срез элементов матрицы по соответствующей оси

        current_square = conj_prod[tuple(square_slices)]  # часть матрицы для обработки
        # цикл по осям, вдоль которых будет происходить смещение спектра
        for shifting_axis in shifting_order:
            contr_shifting_axis = (shifting_axis + 1)  % 2  # индекс оси, вдоль которой не будет происходить смещение спектра на текущей итерации 
            exp_func = exp_row if shifting_axis == 0 else exp_col  # функция формирования матрицы степеней для матрицы центрирования

            shifting_slices[shifting_axis] = slice(0, None)
            power_spectrum = np.zeros(square_shape[shifting_axis])  # накопитель энергетического спектра по компенсируемому напрвлению
            for i in range(square_shape[contr_shifting_axis]):
                shifting_slices[contr_shifting_axis] = i
                power_spectrum += np.power(np.abs(np.fft.fft(current_square[tuple(shifting_slices)])), 2)

            power_spectrum /= square_shape[contr_shifting_axis]
            power_spectrum = np.roll(power_spectrum, power_spectrum.size // 2 - 1)

            filter_window = get_window(('hamming'), power_spectrum.size, fftbins=False)
            power_spectrum *= filter_window

            cs = CubicSpline(np.arange(power_spectrum.size), power_spectrum)
            decimal_indexes = np.linspace(0, power_spectrum.size - 1, power_spectrum.size * freq_mult)
            interpolated_spectrum = cs(decimal_indexes)

            exp_range = range(max_exp-1, max_exp)  # множество степеней для изменения числа узлов сетки
            shift_array = np.zeros(len(exp_range))  # массив значений смещений спектра по компенсируемому направлению
            power_entropy = np.zeros(len(exp_range))  # массив значений энтропии фазы при соответствующих смещениях
            for i, ep in enumerate(exp_range):  # цикл по степеням для изменения размера сетки
                #step = 2 ** (max_exp - 1 - ep)
                max_index = np.argmax(interpolated_spectrum)
                max_index = max_index / interpolated_spectrum.size * square_shape[shifting_axis]
                shift_array[i] = center_indexes[shifting_axis] - max_index
                tmp_square = current_square * np.exp(exp_func(square_shape[0], square_shape[1], shift_array[i]) * 2 * np.pi * 1j)
                hist, _ = np.histogram(np.angle(tmp_square).flatten(), bins='auto')
                hist = hist / tmp_square.size
                power_entropy[i] = entropy(hist)

            ind = np.argmin(power_entropy)
            shift = shift_array[ind]
            current_square *= np.exp(exp_func(square_shape[0], square_shape[1], shift) * 2 * np.pi * 1j)

        # включение обработанного окна в результирующую матрицу
        differ = 0
        # согласование левой границы
        if less_border[1] > 0:
            # срезы по анализируемому окну для согласования
            connection_slices[0] = slice(0, None)
            connection_slices[1] = slice(0, num_eq_layers[1])
            # срезы по результирующей матрицы для согласования
            result_slices[0] = square_slices[0]
            result_slices[1] = slice(less_border[1], less_border[1] + num_eq_layers[1])
            # компенсация смещения значений на участках согласования
            differ = current_square[tuple(connection_slices)] - result[tuple(result_slices)]

        if less_border[0] > 0:
            # срезы по анализируемому окну для согласования
            connection_slices[0] = slice(0, num_eq_layers[0])
            connection_slices[1] = slice(num_eq_layers[1], None) if less_border[1] > 0 else slice(0, None)
            # срезы по результирующей матрицы для согласования
            result_slices[0] = slice(less_border[0], less_border[0] + num_eq_layers[0])
            result_slices[1] = slice(less_border[1] + num_eq_layers[1], more_border[1]) if less_border[1] > 0 else square_slices[1]
            # компенсация смещения значений на участках согласования
            new_differ = current_square[tuple(connection_slices)] - result[tuple(result_slices)]
            if less_border[1] > 0:
                differ = np.concatenate((differ.flatten(), new_differ.flatten()))
            else:
                differ = new_differ.flatten()

        compensation = -np.mean(differ)
        # добавление окна в результирующую матрицу с учетом вычисленной компенсации
        result[tuple(square_slices)] = current_square + compensation
        # перемещение анализируемого окна
        remainder = source_shape - more_border

        less_border[1] = more_border[1] - num_eq_layers[1]
        if remainder [1] > square_width[1]:
            more_border[1] += square_width[1]
        elif remainder [1] > 0:  # следующая полоса - последняя по столбцам
            more_border[1] = source_shape[1]
            square_shape[1] = remainder[1] + num_eq_layers[1]
            center_indexes[1] = square_shape[1] // 2
        else:  # текущая полоса - последняя по столбцам
            less_border[1] = 0
            square_shape[1] = square_width[1] + num_eq_layers[1]
            more_border[1] = square_shape[1]
            center_indexes[1] = square_shape[1] // 2 - 1

            less_border[0] = more_border[0] - num_eq_layers[0]
            if remainder[0] > square_width[0]:
                more_border[0] += square_width[0]
            elif remainder[0] > 0: # следующая полоса - последнияя по строка
                more_border[0] = source_shape[0]
                square_shape[0] = remainder[0] + num_eq_layers[0]
                center_indexes[0] = square_shape[0] // 2 - 1
            else:  # текщая полоса - последняя по строкам
                break

    return result
