"""
    Файл, содержащий функции для фильтрации фазовой составляющей сигнала
"""


import numpy as np
from cv2 import blur, BORDER_REPLICATE 
from scipy.ndimage import median_filter


def get_filtered_phase(method:str, phase_matrix: np.ndarray, kwargs:dict) -> np.ndarray:
    """
    Функция, предоставляющая общий интерфейс обращения к функциям фильтации фазы
    :param method: название метода фильтрации фазы 
                  "average", требует аргументов: windows_step
                  "geom_average", требует аргументов: windows_step
                  "median", требует аргументов: windows_step
    :param phase_matrix: матрица исходной фазы
    :kwargs: словарь аргументов, необходимых для используемого метода
         window_step: радиус окна фильтра
                    нужен для методов "average", "geom_average", "median"
    
    :return: матрица обработанной фазы исходного размера, границы размера меньше радиуса заполняются аналогично BORDER_REPLICATE в opencv
            В случае указания неверного названия матода возвращается None
    """
    
    if method not in methods:
        return None
    else:
        return methods[method](phase_matrix, **kwargs)


def averaging(phase_matrix: np.ndarray, window_step: int) -> np.ndarray:
    """
    Усредняющий фильтр фазы
    :param phase_matrix: матрица исходной фазы
    :param window_step: радиус окна фильтра
    :return: матрица обработанной фазы
    """
    return np.arctan2(blur(np.sin(phase_matrix), (2 * window_step + 1, 2 * window_step + 1), borderType=BORDER_REPLICATE), 
                      blur(np.cos(phase_matrix), (2 * window_step + 1, 2 * window_step + 1), borderType=BORDER_REPLICATE))


def geometric_averaging(phase_matrix: np.ndarray, window_step: int) -> np.ndarray:
    """
    Средне геометрический фильтр фазы
    :param phase_matrix: матрица исходной фазы
    :param window_step: радиус окна фильтра
    :return: матрица обработанной фазы
    """

    return np.arctan2(np.exp(blur(np.log(np.sin(phase_matrix) + 1.+10**(-12)), (2 * window_step + 1, 2 * window_step + 1), borderType=BORDER_REPLICATE)),
                     np.exp(blur(np.log(np.cos(phase_matrix) + 1.+10**(-12)), (2 * window_step + 1, 2 * window_step + 1), borderType=BORDER_REPLICATE)))


def medianing(phase_matrix: np.ndarray, window_step: int) -> np.ndarray:
    """
    Медианный фильтр фазы
    :param phase_matrix: матрица исходной фазы
    :param window_step: радиус окна фильтра
    :return: матрица обработанной фазы
    """
    return np.arctan2(median_filter(np.sin(phase_matrix), size= 2 * window_step + 1, mode="nearest"), 
                      median_filter(np.cos(phase_matrix), size= 2 * window_step + 1, mode="nearest"))

# словарь соответствия названия метода для вызова из общей функции с адресами функций реализаций
methods = {
            "average": averaging,
            "geom_averaging": geometric_averaging,
            "median": medianing
            }
