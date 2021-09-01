"""
    Файл, содержащий функции для построения карт качества матриц кадров
"""


import numpy as np
from sys import float_info
from insar.auxiliary import dispersion_by_window
from cv2 import boxFilter, BORDER_REPLICATE, getStructuringElement, MORPH_RECT, dilate


def get_quality_map(method: str, source_matrix: np.ndarray, kwargs: dict)-> np.ndarray:
    """
    Функция, предоставляющая общий интерфейс обращения к функциям карт качества
    Внимание!!! в тех методах, в которых в названии присутствует слово phase выполняется осуществляется приведение по модулю pi
     в каких-либо мастах вычисления коэффициентов итоговой матрицы
    :param method: название метода фильтрации фазы 
                  "correlation", требует аргументов: second_rli, windows_step
                  "pseudo_correlation", требует аргументов: windows_step
                  "phase_derivative_variance", требует аргументов: windows_step
                  "phase_max_grad", требует аргументов: windows_step
                  "phase_second_diff", не требует аргументов

    :param source_matrix: комплексная матрица рли для метода "correlation"; действительная матрица фазы для остальных методов
    :kwargs: словарь аргументов, необходимых для используемого метода
         second_rli: комплексная матрица второго снимка рли
                    нужен для метода "correlation"
         window_step: радиус окна фильтра
                    нужен для методов "correlation", "pseudo_correlation", "derivative_variance", "max_phase_grad"
    
    :return: матрица качества исходного размера, границы размера меньше радиуса заполняются аналогично BORDER_REPLICATE в opencv
            В случае указания неверного названия матода возвращается None
    """
    if method not in maps:
        return None
    else:
        return maps[method](source_matrix, **kwargs)


def correlation(first_rli: np.ndarray, second_rli: np.ndarray, window_step: int) -> np.ndarray:
    """
    Корреляционая карта строится не только по фазовым данным, а по всему комплексному сигналу.
    :param first_rli: матрица первого комплекснозначного снимка
    :param second_rli: матрица второго комплекснозначного снимка
    :param window_step: радиус окна вычисления значения функции
    :return: матрица качества
    """
    ksize = (2 * window_step + 1,  2 * window_step + 1)  # кортеж размеров ядра фильтра
    numerator = boxFilter(first_rli * np.conj(second_rli), -1, ksize, normalize=False, borderType=BORDER_REPLICATE)  # числитель
    denominator = boxFilter(np.power(np.abs(first_rli), 2), -1, ksize, normalize=False, borderType=BORDER_REPLICATE)  # накопитель знаменателя
    denominator *= boxFilter(np.power(np.abs(second_rli), 2), -1, ksize, normalize=False, borderType=BORDER_REPLICATE)
    np.clip(np.sqrt(denominator), float_info.min, None, out=denominator)
    return numerator / denominator


def pseudo_correlation(wrapped_phase: np.ndarray, window_step: int) -> np.ndarray:
    """
    Псевдокорелляционная карта используется для использования идеи корелляционной карты в том случае,
    когда неизвесны матрицы комплексных значений каждого из двух сигналов в отдельности
    :param phase_matrix: матрица фазы интерферограммы
                        (получена как фаза произведения первого сигнала с сопряженным вторым)
    :param window_step: радиус окна оператора, не считая центральную ячейку
    :return: матрица качества для ячеек, отстоящих от границ матрицы не меньше, чем window_step, по любому измерению
    """
    ksize = (2 * window_step + 1,  2 * window_step + 1)  # кортеж размеров ядра фильтра
    accumulator = np.power(boxFilter(np.sin(wrapped_phase), -1, ksize, normalize=False, borderType=BORDER_REPLICATE), 2)
    accumulator += np.power(boxFilter(np.cos(wrapped_phase), -1, ksize, normalize=False, borderType=BORDER_REPLICATE), 2)
    accumulator = np.sqrt(accumulator) / (2 * window_step + 1) ** 2
    return accumulator


def phase_derivative_variance(wrapped_phase: np.ndarray, window_step: int):
    """
    Функция вычисления матрицы качества по ско частных производных
    :param wrapped_phase: исходная матрица интерферометрической фазы
    :param window_step: радиус окна фильта, не включая центральную яччейку
    :return: матрица качества
    """
    accumulator = np.zeros(wrapped_phase.shape, dtype=wrapped_phase.dtype)
    derivative = wrapped_phase[1:-1, 2:] - wrapped_phase[1:-1, :-2]  # производная по столбцам (горизонтально)
    derivative = np.arctan2(np.sin(derivative), np.cos(derivative))
    accumulator[1:-1, 1:-1] = np.sqrt(dispersion_by_window(derivative, window_step))  # накопитель результата
    derivative = wrapped_phase[2:, 1:-1] - wrapped_phase[:-2, 1:-1]  # производная по строкам (вертикально)
    derivative = np.arctan2(np.sin(derivative), np.cos(derivative))
    accumulator[1:-1, 1:-1] += np.sqrt(dispersion_by_window(derivative, window_step))
    accumulator /= (2 * window_step + 1) ** 2
    return accumulator


def phase_max_grad(wrapped_phase: np.ndarray, window_step: int):
    """
    Функция вычисления матрицы качества по градиенту матрицы фазы интерферограммы
    :param wrapped_phase: исходная матрица интерферометрической фазы
    :param window_step: радиус окна фильта, не включая центральную яччейку
    :return: матрица качества
    """
    ksize = (2 * window_step + 1,  2 * window_step + 1)  # кортеж размеров ядра фильтра
    columns_derivative = wrapped_phase[1:-1, 2:] - wrapped_phase[1:-1, :-2]  # производная по столбцам (горизонтально)
    columns_derivative = np.arctan2(np.sin(columns_derivative), np.cos(columns_derivative))
    rows_derivative = wrapped_phase[2:, 1:-1] - wrapped_phase[:-2, 1:-1]  # производная по строкам (вертикально)
    rows_derivative = np.arctan2(np.sin(rows_derivative), np.cos(rows_derivative))
    accumulator = np.ones(wrapped_phase.shape)
    accumulator *= np.min(np.minimum(columns_derivative, rows_derivative))
    accumulator[1:-1, 1:-1] = np.maximum(columns_derivative, rows_derivative)
    kernel = getStructuringElement(MORPH_RECT, ksize)
    accumulator = -np.maximum(dilate(accumulator, kernel=kernel, borderType=BORDER_REPLICATE), accumulator)
    return accumulator
    

def phase_second_diff(wrapped_phase: np.ndarray) -> np.ndarray:
    """
    Функция вычисления матрицы качества с помощью второй производной по модулю пи
    :param wrapped_phase: исходная матрица интерфероматрической фазы
    :return: матрица качества
    """
    # матрица, которая будет содержать значения функции надёжности
    # граничные пиксели останутся нулями
    rows_num, columns_num = wrapped_phase.shape
    relative_matrix = np.zeros((rows_num, columns_num))

    # учёт второй разности по вертикали окна подсчёта функции надёжности
    tmp_matrix = np.diff(wrapped_phase[:, 1:-1], axis=0)
    tmp_matrix = np.arctan2(np.sin(tmp_matrix), np.cos(tmp_matrix))
    relative_matrix[1:-1, 1:-1] += np.power(np.diff(tmp_matrix, axis=0), 2)

    # учёт второй разности по горизонтали окна подсчёта функции надёжности
    tmp_matrix = np.diff(wrapped_phase[1:-1, :], axis=1)
    tmp_matrix = np.arctan2(np.sin(tmp_matrix), np.cos(tmp_matrix))
    relative_matrix[1:-1, 1:-1] += np.power(np.diff(tmp_matrix, axis=1), 2)

    # учёт второй разности по главной диагонали окна подсчёта функции надёжности
    tmp_matrix = wrapped_phase[:-1, :-1] - wrapped_phase[1:, 1:]
    tmp_matrix = np.arctan2(np.sin(tmp_matrix), np.cos(tmp_matrix))
    relative_matrix[1:-1, 1:-1] += np.power(tmp_matrix[:-1, :-1] - tmp_matrix[1:, 1:], 2)

    # учёт второй разности по побочной диагонали окна подсчёта функции надёжности
    tmp_matrix = wrapped_phase[1:, :-1] - wrapped_phase[:-1, 1:]
    tmp_matrix = np.arctan2(np.sin(tmp_matrix), np.cos(tmp_matrix))
    relative_matrix[1:-1, 1:-1] += np.power(tmp_matrix[1:, :-1] - tmp_matrix[:-1, 1:], 2)
    relative_matrix = np.sqrt(relative_matrix)
    min_val = 0.9 * np.min(relative_matrix[relative_matrix > 0])
    relative_matrix = np.where(relative_matrix < min_val, min_val, relative_matrix)
    relative_matrix = 1 / relative_matrix
    return relative_matrix


# словарь соответствия названия метода для вызова из общей функции с адресами функций реализаций
maps = {
        "correlation": correlation,
        "pseudo_correlation": pseudo_correlation,
        "phase_derivative_variance": phase_derivative_variance,
        "phase_max_grad": phase_max_grad,
        "phase_second_diff":phase_second_diff
        }
