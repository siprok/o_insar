"""
    Скрипт для получения матриц качества и их изображений
    Аргументы:
        folder [str] - название папки, содержащей файл с матрицей, для которой будут вычисляться матрицы качества
        --file [str] - название файла, содержащего матрицу, для которой будут строиться матрицы качества
        --folder_path [str] - путь к папке, указанной в аргументе folder
        --window_step [int] - радиус окна для методов его использующих
        --azimuth_start [int] - индекс начала участка обработки по строкам
        --azimuth_width [int] - ширина участка обработки по строкам (в количестве дискретов)
        --range_start [int] - индекс начала участка обработки по столбцам
        --range_width [int] - ширина участка обработки по столбцам (в количестве дискретов)
    Пример использования:
    python ./make_quality_maps.py ./processing_results/ --file phase.npy  --folder_path /root/user/new_folder/bla_bla/ --window_step 2 --azimuth_start 120 --azimuth_width 500 --range_start 360 --range_width 700
                                  ./processing_results                                  /root/user/new_folder/bla_bla
                                 (допустимы оба варианта)                                   (допустимы оба варианта)                    
"""


import numpy as np
import argparse
from subprocess import Popen
from insar.unwrapping.quality_maps import get_quality_map
from pathlib import Path

if __name__ == '__main__':
    # Создадим  парсер аргументов скрипта
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('folder', help='label folder, which consist source file', type=str)
    parser.add_argument('--file', help='label of source file', type=str)
    parser.add_argument('--folder_path', help='path to source_file', type=str)
    parser.add_argument('--window_step', type=int)
    parser.add_argument('--azimuth_start', type=int)
    parser.add_argument('--azimuth_width', type=int)
    parser.add_argument('--range_start', type=int)
    parser.add_argument('--range_width', type=int)
    # Получим значения аргументов, в случае отсутствия присвим значения по-умолчанию
    args = parser.parse_args()
    folder = args.folder  # название папки, содержащей файл с матрицей, для которой будут вычисляться матрицы качества
    folder = folder.rstrip("/") + "/"  # приведение строки к однму варианту последнего символа для устойчивости дальнейшего использования
    phase_file = args.file or "phase.npy"  # название файла, содержащего матрицу, для которой будут строиться матрицы качества
    folder_path = args.folder_path or "/home/stepan/zpt/interferometry/processing_results/150522_11-13-26/"  # путь к папке, указанной в переменной folder
    folder_path = folder_path.rstrip("/") + "/"  # приведение к единой форме окончания строки
    
    phase = np.load(folder_path + folder + phase_file)

    window_step = args.window_step or 1  # радиус окна для методов его использующих
    
    azimuth_start = args.azimuth_start or 0 # 2500  индекс начала участка обработки по строкам
    azimuth_width = args.azimuth_width or phase.shape[0] # 2000 ширина участка обработки по строкам
    range_start = args.range_start or 0 # 700 индекс начала участка обработки по столбцам
    range_width = args.range_width or phase.shape[1] # 2000 ширина участка обработки по столбцам
    
    phase = phase[azimuth_start: azimuth_start + azimuth_width, range_start: range_start + range_width]

    quality_maps_folder = Path(folder_path + folder + "quality_maps")
    quality_maps_folder.mkdir(exist_ok=True)  

    py_interp = "./env/bin/python3"
    script = "./processing_scripts/npy_trfm_bmp.py" 
    
    def mes_and_img(map_label, path, script_args):
        print(map_label + ".npy ready")
        proc = Popen([py_interp, script, path, *script_args], cwd="/home/stepan/zpt/interferometry/")
        proc.wait()
        print(map_label + " image ready")
    
    maps_labels = ["phase_max_grad", "phase_second_diff", "pseudo_correlation"]
    maps_args_list = [{"window_step": window_step}, dict(), {"window_step": window_step}]
    saving_alg_args = [["--algorithm", "linear"], ["--algorithm", "histeq"], ["--algorithm", "linear"]]

    for i, label in enumerate(maps_labels):
        npy_path = quality_maps_folder / (label + ".npy")
        np.save(npy_path, get_quality_map(label, phase, maps_args_list[i]))
        mes_and_img(label, npy_path, saving_alg_args[i])



