import numpy as np
from scipy.signal.windows import get_window
from pathlib import Path
import subprocess
from insar.auxiliary import find_energy_part
from insar.unwrapping import quality_maps


cwd = Path("/home/stepan/zpt/interferometry")  # рабоча директория скрипта
folder_path = cwd / "processing_results/150522_11-13-26/tests_26_08"  # путь к папке, содержащей результаты взаимодействий с указанными снимками 
target_label = "hamming"  # название папки, в которой будут размещены результаты скрипта
conj_prod = np.load(folder_path / "compl.npy")
interp = str(cwd / "env/bin/python")  # путь к интерпретатору виртуального окружения

(folder_path / target_label).mkdir(exist_ok=True)  # создадим папку для результатов

rows_indxs_mat = np.arange(conj_prod.shape[0]).reshape(-1, 1) @ np.ones((1, conj_prod.shape[1]))  # матрица индексов строк соответствующих элементов
columns_indxs_mat = np.ones((conj_prod.shape[0], 1)) @ np.arange(conj_prod.shape[1]).reshape(1, -1)  # матрица индексов столбцов соответствующих элементов
multiplier_shift = np.power(-1, rows_indxs_mat + columns_indxs_mat)  # матрица сомножитель для центрирования спектра
centred_fft = np.fft.fft2(conj_prod * multiplier_shift)  # центрированный Фурье-образ

thresh_part = 0.3  # граничное значение доли энергии для поиска области интереса в спектре
[filter_row_start, filter_row_end], [filter_col_start, filter_col_end] = find_energy_part(conj_prod, thresh_part)  # получим индексы области, которая будет умножена на окно
filter_window = np.zeros(conj_prod.shape)  # накопитель для матрицы окна

row_base = get_window("hamming", filter_row_end - filter_row_start)  # одномерная основа по строкам для формирования значащей части окна 
col_base = get_window("hamming", filter_col_end - filter_col_start)  # одномерная основа по столбцам для формирования значащей части окна 

filter_window[filter_row_start: filter_row_end, filter_col_start: filter_col_end] =  np.outer(row_base, col_base)  # заполнение ненулевой части окна

ham_conj = np.fft.ifft2(centred_fft * filter_window) * multiplier_shift  # получим матрицу во временной области после умножения на окно

wind_path = folder_path / target_label
maps_path = folder_path / target_label / "quality_maps" 
maps_path.mkdir(exist_ok=True)  # создадим папку под матрицы качества обработанного кадра

np.save(wind_path / "compl.npy", ham_conj)  # сохраним комплексную матрицу резуьлтата
np.save(wind_path / "ampl.npy", np.abs(ham_conj))  # сохраним амплитудную матрицу результата
np.save(wind_path / "phase.npy", np.angle(ham_conj))  # сохраним фазовую матрицу результата
np.save(maps_path / "phase_second_diff.npy", quality_maps.phase_second_diff(np.angle(ham_conj))) # сохраним матрицу качества по формуле второй производной 
np.save(maps_path / "pseudo_corr.npy", quality_maps.pseudo_correlation(np.angle(ham_conj), 5))  # сохраним псевдокорелляционную матрицу качества

# Получим соответствующие изображения 
script = str(cwd /  "processing_scripts/npy_trfm_bmp.py")
args_dict = {
                wind_path: {"ampl": "", "phase": "--algorithm linear"},
                maps_path: {"phase_second_diff": "--algorithm linear", "pseudo_corr": "--algorithm linear"}
            }
for base_path, sub_dict in args_dict.items():
    for label, addition in sub_dict.items():
        arguments = str(base_path / (label + ".npy")) 
        if addition:
            proc_list = [interp, script, arguments, *(addition.split(" "))]
        else:
            proc_list = [interp, script, arguments]
        proc = subprocess.Popen(proc_list)  
        proc.wait()
        print(label + " image ready")
