"""
    Файл, содержащий функции для развертки фазы сигнала
"""


import numpy as np
from numpy import pi
from time import time
import operator
from progress.bar import IncrementalBar
from insar.auxiliary import min_span_transitions_matrix_cell, pi_mult


def path_independence_test(source_matrix: np.ndarray) -> bool:
    """
    функция проверки на возможность применения зависимых от пути развёртывания алгоритмов
    :param source_matrix: исходная матрица
    :return: true - в случае если нет зависимости от пути развёртывания
            false - в случае если есть зависимость от пути развёртывания
    """
    der_x_y = np.diff(source_matrix, axis=1)
    der_x_y = np.diff(der_x_y, axis=0)
    der_y_x = np.diff(source_matrix, axis=0)
    der_y_x = np.diff(der_y_x, axis=1)
    return (der_x_y == der_y_x).all()


def itohs_unwrapping(source_array: np.ndarray) -> np.ndarray:
    """
    функция одномерной развёртки фазы
    :param source_array: исходный массив
    """
    unwrapped = source_array.copy()
    if len(source_array.shape) > 1:
        rows, columns = source_array.shape
        unwrapped = np.ravel(unwrapped)
    phase_difference = unwrapped[1:] - unwrapped[:-1]
    phase_difference = np.arctan2(np.sin(phase_difference), np.cos(phase_difference))
    phase_difference[0] += unwrapped[0]
    unwrapped[1:] = phase_difference.cumsum()
    if len(source_array.shape) > 1:
        unwrapped = np.reshape(unwrapped, (rows, columns))
    return unwrapped


def td_itohs_unwrapping(wrapped_phase: np.ndarray) -> np.ndarray:
    """
    Функция двумерной развёртки фазы.
    Алгоритм является зависящим от пути (path-dependence)
    Применяется, если справедливы предположения:
        - в абсолютной фазе между значениями в соседних ячейках нет разности по модулю большей пи
        - смешанные производные исходной матрицы равны
    :param source_array: двумерная матрица с исходной фазой
    :return: двумерная матрица, содержащая абсолютую фазу
    """
    unwrapped = wrapped_phase.copy()
    unwrapped[:, 0] = itohs_unwrapping(wrapped_phase[:, 0])
    for row in range(1, unwrapped.shape[0]):
        unwrapped[row, :] = itohs_unwrapping(unwrapped[row, :])
    return unwrapped


def relnp(wrapped_phase: np.ndarray, relative_matrix: np.ndarray) -> np.ndarray:
    """
        Fast two-dimensional phase-unwrapping algorithm
    based on sorting by reliability following a noncontinuous path
    DOI: 10.134/AO.41.007437
    :param wrapped_phase: матрица свёрнутой фазы
    :param relative_matrix: матрица качества для матрицы фазы
    :return: абсолютная фаза
    """

    def finished(wrapped: np.ndarray, sorted_edges: np.ndarray) -> np.ndarray:
        """
        Фунцкия непосредственно развёртки фазы по подготовленным данным
        :param wrapped: wrapped phase matrix
        :param sorted_edges: sorted array of vertex identifying edges
        :return: unwrapped phase matrix
        """
        bar = IncrementalBar('Edges completed', max=sorted_edges.shape[0], suffix="%(percent)f%%")
        unwrapped_phase = wrapped.copy().flatten()

        # ключ - номер компоненты связности,
        # значение - добавочный множитель для развёртки, количество элементов в компоненте связности
        adjacent_components = dict()
        number_edges = unwrapped_phase.size

        nodes = np.arange(number_edges + 1)

        for edge in sorted_edges:
            first_node = edge[0]  # индекс первого элемента
            first_value = unwrapped_phase[first_node]  # значение фазы в ячейке первого элемента
            first_num = nodes[first_node]  # номер компоненты связности, которой принадлежит первый элемент
            second_node = edge[1]  # индекс второго элемента
            second_value = unwrapped_phase[second_node]  # занчение фазы в ячейке второго элемента
            second_num = nodes[second_node]  # номер компоненты связности, которой принадлежит второй элемент

            components_flag = 0  # переменная хранящая код типов компонент связности анализируемых вершин:
            # 00 - обе компоненты одноэлементны
            # 10 - в первой компоненте более одной вершины, во второй компоненте - одна
            # 01 - в первой компоненте - одна вершина, во второй компоненте более одной вершины
            # 11 - во обеих компонентах более одной вершины

            # получение информации о числе вершин в первой компоненте связности
            if first_num in adjacent_components.keys():
                components_flag += 10
            # получение информации о числе вершин во второй компоненте связности
            if second_num in adjacent_components.keys():
                components_flag += 1

            # случай, когда обе компоненты связности содержат более одной вершины
            if components_flag == 11:
                first_component = adjacent_components[first_num]  # множество индесков ячеек, принадлежащих компоненте связности, в которой лежит первый элемент
                second_component = adjacent_components[second_num]  # множество индесков ячеек, принадлежащих компоненте связности, в которой лежит второй элемент
                if len(first_component) > len(second_component): # в первой компоненте связности вершин больше, чем во второй
                    del adjacent_components[second_num]  # удаляем малую компоненту связности из словаря
                    adjacent_components[first_num] = adjacent_components[first_num].union(second_component)  # объединяем компоненты связности
                    diff = first_value - second_value  # находим разность значений фазы в вершинах рассматриваемого ребра
                    if np.abs(diff) > pi:
                        unwrapped_phase[list(second_component)] += 2 * pi * pi_mult(diff) # компенсация разрыва фазы
                    nodes[list(second_component)] = first_num  # установка принадлежности элементов второй компоненты свяности к первой
                else:  # во второй компоненте связности вершин не меньше, чем в первой
                    del adjacent_components[first_num]   # удаляем малую компоненту связности из словаря
                    adjacent_components[second_num] = adjacent_components[second_num].union(first_component)  # объединяем компоненты связности
                    #diff = first_value - second_value  # находим разность значений фазы в вершинах рассматриваемого ребра
                    diff = second_value - first_value  # находим разность значений фазы в вершинах рассматриваемого ребра
                    if np.abs(diff) > pi:
                        #unwrapped_phase[list(first_component)] += second_value + np.arctan2(np.tan(diff)) - first_value  # компенсация разрыва фазы
                        unwrapped_phase[list(first_component)] += 2 * pi * pi_mult(diff)  # компенсация разрыва фазы
                    nodes[list(first_component)] = second_num  # установка принадлежности элементов первой компоненты свяности ко второй

            elif components_flag == 10:  # в первой компоненте более одной вершины, во второй компоненте - одна
                diff = first_value - second_value  # находим разность значений фазы в вершинах рассматриваемого ребра
                if np.abs(diff) > pi:
                    unwrapped_phase[second_node] += 2 * pi * pi_mult(diff)   # компенсация разрыва фазы
                adjacent_components[first_num].add(second_node)  # добавление второй ячейки к первой компоненте связности
                nodes[second_node] = first_num  # установка принадлежности второй ячейки свяности к первой компоненте связности

            else:
                diff = second_value - first_value  # находим разность значений фазы в вершинах рассматриваемого ребра
                if np.abs(diff) > pi:
                    unwrapped_phase[first_node] += 2 * pi * pi_mult(diff)    # компенсация разрыва фазы
                if components_flag == 1:  # в первой компоненте - одна вершина, во второй компоненте более одной вершины
                    adjacent_components[second_num].add(first_node)  # добавление первой ячейки ко второй компоненте связности
                else:  # обе компоненты одноэлементны
                    adjacent_components[second_node] = {first_node, second_node}  # добавление компоненты свзности в словарь
                nodes[first_node] = second_num  # установка принадлежности первой ячейки свяности ко второй компоненте связности
            bar.next()
        bar.finish()
        del sorted_edges
        del adjacent_components
        print(np.max(unwrapped_phase))
        print(np.min(unwrapped_phase))
        return unwrapped_phase.reshape(wrapped.shape)

    # так как в дальнейшем используется функция построения минимального остова, а развёртывать фазу необходимо
    # по ячейкам с наибольшими значениями матрицы качества, то необходимо выполнить операцию обращающую порядок
    # элементов в матрице
    tmp_relative_matrix = relative_matrix.copy()
    rel_min = np.min(tmp_relative_matrix)
    if rel_min == 0:
        tmp_relative_matrix += 1
    else:
        tmp_relative_matrix -= rel_min - 1  # добавляем единицу для того, чтобы можно было обращать
    tmp_relative_matrix = 1 / tmp_relative_matrix

    start = time()
    adj_matrix_mst = min_span_transitions_matrix_cell(tmp_relative_matrix)
    end = time()
    del tmp_relative_matrix

    print("Algorithm before sorting need {} seconds".format(end - start))
    # получим сортированный по возрастанию список рёбер, заданный номерами узлов(индексами ячеек матрицы),
    # в данном порядке будем производить развёртку фазы
    start = time()
    sorted_edges = np.array(sorted(adj_matrix_mst.items(), reverse=True, key=operator.itemgetter(1)), dtype=object)[:, 0]
    end = time()
    print('Sorting need {} seconds'.format(end - start))

    del adj_matrix_mst

    start = time()
    f = finished(wrapped_phase, sorted_edges)
    end = time()
    print('Finish unwrapping need {}'.format(end - start))
    return f
    