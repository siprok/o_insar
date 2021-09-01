"""
    Файл, содержащий функции чтения/записи информации установленного формата
"""


import numpy as np
from insar.zpt_scripts.hlgio import read_hlg_chunk, HlgChunkConstraints, read_header, get_channel_count


def get_from_rli(source_path: str, data_type="complex", azimuth_start=0, azimuth_width=-1, range_start=0, range_width=-1) -> np.ndarray:
    """
    :param source_path: путь к rli файлу
    :param data_type: желаемый тип информации варианты "complex", "amplitude", "phase" 
    :param azimuth_start: индекс строки, с которой начнётся извлечение куска рли
    :param azimuth_width: число строк, которые будут извлечены из рли
    :param range_start: индекс столбца с которого начнётся извлечение куска рли
    :param range_width: число столбцов, которые будут извлечены из рли
    :return: np.ndarray, матрица, содержащая фазу, указанного куска рли
    """
    (chunk, _, header, _) = read_hlg_chunk(source_path, HlgChunkConstraints(aStart=azimuth_start,
                                                                                              aWidth=azimuth_width,
                                                                                              rStart=range_start,
                                                                                              rWidth=range_width))
    header = read_header(source_path)
    if data_type == 'amplitude':
        chunk = np.abs(chunk)
    else:
        if get_channel_count(header) == 1:
            raise ValueError('rli has only one channel')
        if data_type == 'complex':
            pass
        elif data_type == 'phase':
            chunk = np.angle(chunk)

    return chunk