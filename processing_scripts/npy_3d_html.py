"""
    Скрипт для сохранения отображения значений матрицы в виде поверхности
    Аргументы:
        source_path [str] - путь к исходному файлу
        --destination_path [str]- путь к файлу,в который будет сохранен результат        
        --azimuth_start [int] - индекс начала участка обработки по строкам
        --azimuth_width [int] - ширина участка обработки по строкам (в количестве дискретов)
        --range_start [int] - индекс начала участка обработки по столбцам
        --range_width [int] - ширина участка обработки по столбцам (в количестве дискретов)
    Пример использования:
    python ./npy_3d_html.py ./processing_sources/phase.npy --destination_path ./processing_results/unwrapped_phase.npy --azimuth_start 120 --azimuth_width 500 --range_start 360 --range_width 700                                 
"""


import numpy as np
import plotly.graph_objects as go
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ploting surface by height in npy ')
    parser.add_argument('source_path', type=str)
    parser.add_argument('--destination_path', type=str)    
    parser.add_argument('--azimuth_start', type=int)
    parser.add_argument('--azimuth_width', type=int)
    parser.add_argument('--range_start', type=int)
    parser.add_argument('--range_width', type=int)
    args = parser.parse_args()

    destination = args.destination_path or os.path.splitext(args.source_path)[0] + ".html"
    z_data = np.load(args.source_path)
    rows_start = args.azimuth_start or 0
    rows_stop = args.azimuth_width or z_data.shape[0]
    rows_stop += rows_start

    columns_start = args.range_start or 0
    columns_stop = args.range_width or z_data.shape[1]
    columns_stop += columns_start

    fig = go.Figure(data=[go.Surface(z=np.fliplr(z_data[rows_start:rows_stop, columns_start:columns_stop]))])

    fig.write_html(destination)

