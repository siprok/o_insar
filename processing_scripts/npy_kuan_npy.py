"""
    Скрипт для фильтрации амплитудной/мощностной части сигнала и сохранения результата в npy файл
    Аргументы:
    source_path [str] - путь к исходному файлу
    --destination_path [str]- путь к файлу,в который будет сохранен результат
    --window_step [int] - радиус окна фильтрации
    --enl [int] - количетво доступных кадров наблюдения
    --rli_type [int] - тип кадра (амплитудный или мощностной)
    Пример использования:
    python ./npy_kuan_npy.py  /root/user/new_folder/bla_bla.npy --destination_path /root/user/new_folder/bla_bla_kuan.npy --window_step 2 --enl 7 --rli_type power                                                          
"""


import numpy as np
import argparse
from time import time
from insar.filtering import amplitude


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filtering with Kuan's algorthm")
    parser.add_argument('source_path', type=str, help='Path to source file')
    parser.add_argument('destination_path', type=str, help='Path to save filtered file')
    parser.add_argument('--window_step', type=int, help='Radius filter window')
    parser.add_argument('--enl', type=int)
    parser.add_argument('--rli_type', type=str)
    args = parser.parse_args()

    chunk = np.load(args.source_path).astype(np.float64)
    ws = args.window_step or 7
    enl = args.enl or 1
    rli_type = args.rli_type or 'amplitude'

    amplitude.kuan_argument_checker(chunk, ws, enl, rli_type)
    start_time = time()
    result = amplitude.kuan_filter(chunk, ws, enl, rli_type)
    end_time = time()

    print('Время фильтрации: ', end_time - start_time)
    np.save(args.destination_path, result)
