"""
    Скрипт запуска последовательности обработки данных на разных наборах параметров
    Основная цель - визуальная оценка качества обработки при различных наборах значений параметров
    компенсация набега -> формирование и сохранение карт качества -> развертка и сохранения результата
    Пример использования:
    python ./params_test.py
"""


import numpy as np
import subprocess
from insar.unwrapping import quality_maps
from itertools import product
import os

cwd = "/home/stepan/zpt/interferometry"
py_interp = "./env/bin/python"
folder_path = "./processing_results/150522_11-13-26/tests_26_08/hamming/"
file = "compl.npy"
compl = np.load(folder_path + file)

shifting_axis = "range"
mode = "single"
counter = 0
for az_stripe_width in [1000, 2000]:
    proc_list = []
    for rng_stripe_width in [50, 100, 200]:
        # компенсация набега фазы по полосам
        folder = "width_az_" + str(az_stripe_width) + "_range_" + str(rng_stripe_width) 
        print("\n"+folder)
        
        script = "./processing_scripts/compensating.py"
        script_args =   folder + \
                        " --file " + file + \
                        " --folder_path " + folder_path + \
                        " --az_stripe_width " + str(az_stripe_width) + \
                        " --rng_stripe_width " + str(rng_stripe_width) + \
                        " --shifting_axis " + shifting_axis + \
                        " --mode " + mode 

        if mode == "single":
            strip_proc = subprocess.Popen([py_interp, script, *(script_args.split(" "))], cwd=cwd)
            strip_proc.wait()

            # Карты качества
            script = "./processing_scripts/make_quality_maps.py"
            script_args = folder + \
                          " --file strip_phase.npy" + \
                          " --folder_path " + folder_path
            maps_proc = subprocess.Popen([py_interp, script, *(script_args.split(" "))], cwd=cwd)
            maps_proc.wait()
            
            # Развертка
            script = "./processing_scripts/unwrapping.py"
            script_args =   folder + \
                            " --file strip_phase.npy" +\
                            " --folder_path " + folder_path + \
                            " --algorithm relnp"

            unw_proc = subprocess.Popen([py_interp, script, *(script_args.split(" "))], cwd=cwd)
            unw_proc.wait()
            
        else:
            proc_list.append(subprocess.Popen([py_interp, script, *(script_args.split(" "))], cwd=cwd))
            
    for p in proc_list:
        p.wait()

    
