"""
    Скрипт для сохранения РЛИ в формате npy
    Аргументы:
        source_path [str] - путь к исходному файлу
        destination_path [str]- путь к файлу,в который будет сохранен результат 
        --data_type [str] - тип сигнада, который нужно извлечь (комплексный, амплитудный, фазовый)       
        --azimuth_start [int] - индекс начала участка обработки по строкам
        --azimuth_width [int] - ширина участка обработки по строкам (в количестве дискретов)
        --range_start [int] - индекс начала участка обработки по столбцам
        --range_width [int] - ширина участка обработки по столбцам (в количестве дискретов)
    Пример использования:
    python ./rli_npy.py ./processing_sources/source.rli destination_path ./processing_results/phase.npy --data_type phase --azimuth_start 120 --azimuth_width 500 --range_start 360 --range_width 700                                 
"""


import numpy as np
import argparse
from insar.zpt_scripts import hlgio

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converting rli to file containing filtered matrix')
    parser.add_argument('source_path', type=str, help='Path to rli file')
    parser.add_argument('destination_path', type=str, help='Path to save filtered file')
    parser.add_argument('--data_type', type=str, help='Type info which you to get (complex(default), amplitude, phase)')
    parser.add_argument('--azimuth_start', type=int)
    parser.add_argument('--azimuth_width', type=int)
    parser.add_argument('--range_start', type=int)
    parser.add_argument('--range_width', type=int)

    args = parser.parse_args()

    header = hlgio.read_header(args.source_path)
    (chunk, hlgSize, header, _) = hlgio.read_hlg_chunk(args.source_path, hlgio.HlgChunkConstraints(
                                                                                       rWidth=args.range_width or -1,
                                                                                       rStart=args.range_start or 0,
                                                                                       aWidth=args.azimuth_width or -1,
                                                                                       aStart=args.azimuth_start or 0))
    if not args.data_type or args.data_type == 'complex':
        np.save(args.destination_path, chunk)
    elif args.data_type == 'amplitude':
        np.save(args.destination_path, np.abs(chunk))
    elif args.data_type == 'phase':
        if hlgio.get_channel_count(header) == 2:
            chunk = np.angle(chunk)
            np.save(args.destination_path, chunk)
        else:
            print("Have no phase component")

