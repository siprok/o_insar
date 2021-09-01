"""
    Скрипт для компенсации набега фазы и сохранения результата в виде файлов npy и изображений
    Аргументы:
        folder [str] - название папки, содержащей файл с матрицей
        --file [str] - название файла 
        --folder_path [str] - путь к папке
        --az_stripe_width [int] - ширина окна по строкам
        --rng_stripe_width [int] - ширина окна по столбцам
        --shifting_axis [str] - ось вдоль которой будет происходить сдвиг (по азимуту или дальности)
        --td_flag - флаг проведения компенсации по лвум осям
        --mode [str] - тип сохранения данных результатов
        --azimuth_start [int] - индекс начала участка обработки по строкам
        --azimuth_width [int] - ширина участка обработки по строкам (в количестве дискретов)
        --range_start [int] - индекс начала участка обработки по столбцам
        --range_width [int] - ширина участка обработки по столбцам (в количестве дискретов)
    Пример использования:
    python ./compensating.py processing_sources --file complex.npy --folder_path /root/user/bla_bla --az_stripe_width 2000 --rng_stripe_width 30 --shifting_axis range --mode single --azimuth_start 120 --azimuth_width 2000 --range_start 360 --range_width 700                                 
"""


import numpy as np
import subprocess
from argparse import ArgumentParser
from insar.compensate import grid_shift_interpolate
import os


if __name__ == '__main__':
    parser = ArgumentParser(description='Shifting each strip')
    parser.add_argument('folder', type=str)
    parser.add_argument('--file', help='label of source file', type=str)
    parser.add_argument('--folder_path', help='path to source_file', type=str)
    parser.add_argument('--az_stripe_width', type=int)
    parser.add_argument('--rng_stripe_width', type=int)
    parser.add_argument('--shifting_axis', help='azimuth, range', type=str)
    parser.add_argument('--td_flag', dest="td_flag", action="store_true")
    parser.set_defaults(td_flag=False)
    parser.add_argument('--mode', help='multy or single', type=str)
    parser.add_argument('--azimuth_start', type=int)
    parser.add_argument('--azimuth_width', type=int)
    parser.add_argument('--range_start', type=int)
    parser.add_argument('--range_width', type=int)

    axis_map = {"azimuth": 0, "range": 1}

    args = parser.parse_args()
    folder = args.folder  # название папки, содержащей файл с матрицей
    folder = folder.rstrip("/") + "/"  # приведение строки к однму варианту последнего символа для устойчивости дальнейшего использования
    compl_file = args.file or "compl.npy"
    folder_path = args.folder_path or "/home/stepan/zpt/interferometry/processing_results/150522_11-13-26/"  # путь к папке, указанной в переменной folder
    folder_path = folder_path.rstrip("/\\") + "/"
    az_stripe_width = args.az_stripe_width or 500 # количство строк в окне обработки
    if az_stripe_width % 2 == 1:
        az_stripe_width -= 1  
    rng_stripe_width = args.rng_stripe_width or 500  # коичство столбцов в окне обработки
    if rng_stripe_width % 2 == 1:
        rng_stripe_width -= 1
    shifting_axis = axis_map[args.shifting_axis] if args.shifting_axis else axis_map["range"]
    td_flag = args.td_flag
    azimuth_start = args.azimuth_start or 2500
    azimuth_width = args.azimuth_width or 2000
    range_start = args.range_start or 700
    range_width = args.range_width or 2000
    
    mode = args.mode or "single"  # выбор режима 
                                # single - будут сохранены изображения и матрицы в формате npy. Набор данных для каждой комбинации параметров сохраняется в своей папке
                                # multy - будут сохранены только изображения результатов обработки. Все изображения сохраня.тся в одной папке, в названи каждого файла соержится набор параметров 

    if mode == "single":
        try:
            os.makedirs(folder_path + folder)
        except FileExistsError:
            pass
    
    else:
        ampl_path = folder_path + "striped_ampl/"
        phase_path = folder_path + "striped_phase/"
        try:
            os.makedirs(ampl_path)
        except FileExistsError:
            pass   

        try:
            os.makedirs(phase_path)
        except FileExistsError:
            pass 
    
    striped = np.load(folder_path + compl_file)[azimuth_start: azimuth_start + azimuth_width, range_start: range_start + range_width]
    if striped.shape[0] % 2 == 1:
        striped = striped[:-1,:]
    if striped.shape[1] % 2 == 1:
        striped = striped[:,:-1]

    striped = grid_shift_interpolate(striped, az_stripe_width, rng_stripe_width, 0.6, 3, shifting_axis, td_flag)

    py_interp = "./env/bin/python"
    script = "./processing_scripts/npy_trfm_bmp.py" 
    if mode == "single":
        folder_path += folder
        np.save(folder_path + "strip_compl.npy", striped)
        print("compl.npy ready")

        npy_path = folder_path + "strip_ampl.npy"
        script_args = npy_path
        np.save(npy_path, np.abs(striped))
        print("ampl.npy ready")

        proc = subprocess.Popen([py_interp, script, *(script_args.split(" "))], cwd="/home/stepan/zpt/interferometry/")
        proc.wait()
        print("ampl.bmp ready")

        npy_path = folder_path + "strip_phase.npy"
        script_args = npy_path + " --algorithm linear"
        np.save(npy_path, np.angle(striped))
        print("phase.npy ready")
        proc = subprocess.Popen([py_interp, script, *(script_args.split(" "))], 
                                cwd="/home/stepan/zpt/interferometry/")
        proc.wait()
        print("phase.bmp ready")
    else:
        destination = folder[:-1].split("\\")
        dir = "\\".join(destination[:-1])
        try:
            os.makedirs(ampl_path + dir)
        except FileExistsError:
            pass 
        npy_path = ampl_path + folder[:-1] + ".npy"
        script_args = npy_path
        np.save(npy_path, np.abs(striped))

        proc = subprocess.Popen([py_interp, script, *(script_args.split(" "))],
                                cwd="/home/stepan/zpt/interferometry/")
        proc.wait()
        os.remove(npy_path)
        print(destination[-1] + " amplitude ready")

        try:
            os.makedirs(phase_path + dir)
        except FileExistsError:
            pass 
        npy_path = phase_path + folder[:-1] + ".npy"
        script_args = npy_path + " --algorithm linear"
        np.save(npy_path, np.angle(striped))
        proc = subprocess.Popen([py_interp, script, *(script_args.split(" "))],
                                cwd="/home/stepan/zpt/interferometry/")
        proc.wait()
        os.remove(npy_path)
        print(destination[-1] + " phase ready")
