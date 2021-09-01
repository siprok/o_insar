"""
    Скрипт для развертки фазы с различными картами качества и сохранения результата в виде npy файла и полутонового изображения
    Аргументы:
        folder [str] - название папки, содержащей файл с матрицей
        --file [str] - название файла 
        --folder_path [str] - путь к папке
        --algorithm [str] - способ развертки фазы
        --azimuth_start [int] - индекс начала участка обработки по строкам
        --azimuth_width [int] - ширина участка обработки по строкам (в количестве дискретов)
        --range_start [int] - индекс начала участка обработки по столбцам
        --range_width [int] - ширина участка обработки по столбцам (в количестве дискретов)
    Пример использования:
    python ./unwrapping.py processing_sources --file phase.npy --folder_path /home/user/processing_results/ --algorithm relnp --quality_map sd --azimuth_start 120 --azimuth_width 500 --range_start 360 --range_width 700                                 
                           processing_sources/                               /home/user/processing_results
                          \__________________/                               \____________________________/
                        (допустимы оба варианта)                                (допустимы оба варианта)
"""


import argparse
import subprocess
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unwrapping phase matrix with pcm, sd, mpg quality maps')
    parser.add_argument('folder', type=str)
    parser.add_argument('--file', help='label of source file', type=str)
    parser.add_argument('--folder_path', help='path to source_file', type=str)
    parser.add_argument('--algorithm', help='relnp, itohs, td_itohs', type=str)
    parser.add_argument('--azimuth_start', type=int)
    parser.add_argument('--azimuth_width', type=int)
    parser.add_argument('--range_start', type=int)
    parser.add_argument('--range_width', type=int)

    args = parser.parse_args()
    folder = args.folder
    folder = folder.rstrip("/") + "/"
    phase_file = args.file or "phase.npy"
    folder_path = args.folder_path or "./processing_results/150522_11-13-26/"
    folder_path = folder_path.rstrip("/") + "/"
    if args.azimuth_start == 0: 
        azimuth_start = 0  # индекс начала участка обработки по строкам 
    else:
        azimuth_start = args.azimuth_start or 0  # 2500 индекс начала участка обработки по строкам 

    if args.azimuth_width == 0:
        azimuth_start = 0
    else:
        azimuth_width = args.azimuth_width or 500  # ширина участка обработки по строкам

    if args.range_start == 0:
        range_start = 0  # индекс начала участка обработки по столбцам
    else:
        range_start = args.range_start or 0  # 700 индекс начала участка обработки по столбцам

    if args.range_width == 0:
        range_width = 0  # ширина участка обработки по столбцам
    else:
        range_width = args.range_width or 500  # ширина участка обработки по столбцам

    py_interp = "./env/bin/python"

    try:
        os.makedirs(folder_path + folder)
    except FileExistsError:
        pass  

    cwd = "/home/stepan/zpt/interferometry"
    script_unw = "./processing_scripts/npy_unwrap_npy.py"
    script_unw_args = folder_path + folder + phase_file +\
                      " --azimuth_start " + str(azimuth_start) +\
                      " --azimuth_width " + str(azimuth_width) +\
                      " --range_start " + str(range_start) +\
                      " --range_width " + str(range_width) +\
                      " --quality_map" 

    script_img = "./processing_scripts/npy_trfm_bmp.py"
    script_img_args = folder_path + folder + phase_file[:-4] + "_unwrapped_relnp_" 

    # Развертка
    for qm in ["sd", "pcm", "mpg"]:
        proc = subprocess.Popen([py_interp, script_unw, *(script_unw_args.split(" ")), qm] , cwd=cwd)
        proc.wait()
        print(phase_file[:-4] + "_unwrapped_relnp_" + qm + ".npy ready")

        proc = subprocess.Popen([py_interp, script_img, script_img_args + qm + ".npy", "--algorithm", "linear"], cwd=cwd)
        proc.wait()
        print(phase_file[:-4] + "_unwrapped_relnp_" + qm + ".bmp ready")

