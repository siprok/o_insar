"""
    Скрипт для фильтрации фазы всеми способами, описанными в filtering/phase.py и сохранения результата в виде файлов npy и изображений
    Аргументы:
        folder [str] - название папки, содержащей файл с матрицей
        --file [str] - название файла 
        --folder_path [str] - путь к папке
        --azimuth_start [int] - индекс начала участка обработки по строкам
        --azimuth_width [int] - ширина участка обработки по строкам (в количестве дискретов)
        --range_start [int] - индекс начала участка обработки по столбцам
        --range_width [int] - ширина участка обработки по столбцам (в количестве дискретов)
    Пример использования:
    python ./phase_filtering.py processing_sources --file phase.npy --folder_path /root/user/bla_bla --azimuth_start 120 --azimuth_width 2000 --range_start 360 --range_width 700                                 
"""


import numpy as np
import argparse
from subprocess import Popen
from insar.filtering.phase import get_filtered_phase
from insar.filtering.phase import methods
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('folder', help='label folder, which consist source file', type=str)
    parser.add_argument('--file', help='label of source file', type=str)
    parser.add_argument('--folder_path', help='path to source_file', type=str)
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
    
    azimuth_start = args.azimuth_start or 0 # 2500  индекс начала участка обработки по строкам
    azimuth_width = args.azimuth_width or phase.shape[0] # 2000 индекс конца участка обработки по строкам
    range_start = args.range_start or 0 # 700 индекс начала участка обработки по столбцам
    range_width = args.range_width or phase.shape[1] # 2000 индекс конца участка обработки по столбцам
    
    phase = phase[azimuth_start: azimuth_start + azimuth_width, range_start: range_start + range_width]

    filtered_phase_folder = Path(folder_path + folder + "filtered_phase")
    filtered_phase_folder.mkdir(exist_ok=True)  

    py_interp = "./env/bin/python3"
    script = "./processing_scripts/npy_trfm_bmp.py" 
    
    def mes_and_img(map_label, path, script_args):
        print(map_label + ".npy ready")
        proc = Popen([py_interp, script, path, *script_args], cwd="/home/stepan/zpt/interferometry/")
        proc.wait()
        print(map_label + ".bmp ready")
    
    
    saving_alg_args = [["--algorithm", "linear"]] * 3

    for i, label in enumerate(methods):
        for w_s in range(1,5):
            npy_path = str(filtered_phase_folder) + "/" + label + "_" + str(w_s) + ".npy" 
            np.save(npy_path, get_filtered_phase(label, phase, {"window_step": w_s}))
            mes_and_img(label, npy_path, saving_alg_args[i])



