# InSAR
## Содержание <a name="content"></a>
1. [Содержание](#content)
2. [Краткое описание](#summary)
3. [Используемые сторонние библиотеки](#libs)
4. [Структура репозитория](#structure)
5. [Незавершенные этапы обработки](#unfinished)
## Краткое описание <a name="summary"> </a>
Создание цифровой модели рельефа по радиолокационным интерферометрическим снимкам с некогерентным накоплением.
<p> Пример обработки данных </p>

| :Способ отображения/Этап обработки: | :Фаза до развертки: | :Фаза после развертки: |
| :Полутоновое изображение: | ![2d wrapped phase](https://gitlab.com/siprok/interferometry/examples/2d_wrapped_phase.png) | ![2d unwrapped phase](https://gitlab.com/siprok/interferometry/examples/2d_unwrapped_phase.png) |
| :Поверхность: | ![3d wrapped phase](https://gitlab.com/siprok/interferometry/examples/3d_wrapped_phase.png) | ![3d unwrapped phase](https://gitlab.com/siprok/interferometry/examples/3d_unwrapped_phase.png) |

## Используемые сторонние библиотеки <a name="libs"> </a>
<p>scipy, progress, opencv-python </p>
<p> После установки сторонних библиотек и клонирования репозитория, можно запускать скрипты с помощью интерпретатора python3.</p>

## Структура репозитория <a name="structure"> </a>
<ul>
<li> insar | Папка, содержащая функции для обработки данных
  <ul>
    <li> filtering | папка, содержащая файлы с функциями для фильтрации сигнала
      <ul>
         <li> complex.py   | Файл, содержащий функции для фильтрации комплексного сигнала </li>
         <li> amplitude.py | Файл, содержащий функции для фильтрации амплитудной составляющей сигнала </li>
         <li> phase.py     | Файл, содержащий функции для фильтрации фазовой составляющей сигнала </li>
      </ul>
    </li>
    <li> unwrapping | папка, содержащая файлы с функциями для развертки фазы сигнала
      <ul>
         <li> algorithms.py   | Файл, содержащий функции для развертки фазы сигнала </li>
         <li> masks.py        | Файл, содержащий функции для построения логических масок по матрицам сигналов </li>
         <li> quality_maps.py | Файл, содержащий функции для построения карт качества матриц кадров </li>
      </ul>
    </li>
    <li> zpt_scripts | папка, содержащая файлы с функциями для извлечения данных из hlg файлов
      <ul>
        <li> hlgio.py </li>
        <li> ph.py </li>
      </ul>
    </li>
    <li> alignment.py | Файл, содержащий функции для совмещения РЛИ </li>
    <li> auxiliary.py | Файл, содержащий вспомогательные функции  </li>
    <li> compensate.py | Файл, содержащий функции компенсации фазовых набегов </li>
    <li> data_io.py | Файл, содержащий функции чтения/записи информации установленного формата </li>
  </ul>
</li>
<li> processing_scripts | Папка, содержащая скрипты, использующие функции из insar для обработки данных из терминала
   <ul>
      <li>compensating.py | Скрипт для компенсации набега фазы и сохранения результата в виде файлов npy и изображений </li>
      <li>make_quality_maps.py | Скрипт для получения матриц качества и их изображений </li>
      <li>npy_3d_htnl.py | Скрипт для сохранения отображения значений матрицы в виде поверхности) </li>
      <li>npy_kuan_npy.py | Скрипт для фильтрации амплитудной/мощностной части сигнала и сохранения результата в npy файл </li>
      <li>npy_trfm_bmp.py | Скрипт для сохранения отображения значений матрицы в виде полутонового изображения </li>
      <li>npy_unwrap_npy.py | Скрипт для развертки фазы и сохранения результата в виде npy файла </li>
      <li>params_test.py | Скрипт запуска последовательности обработки данных на разных наборах параметров </li>
      <li>phase_filtering.py | Скрипт для фильтрации фазы всеми способами, описанными в filtering/phase.py и сохранения результата в виде файлов npy и изображений</li>
      <li> rli_npy.py | Скрипт для сохранения РЛИ в формате npy</li>
      <li> unwrapping.py | Скрипт для развертки фазы с различными картами качества и сохранения результата в виде npy файла и полутонового изображения </li>
   </ul>
</li>
</ul>

<p> Вычислительные модули расположены в папке insar. Скрипты, использующие вычислитльные модули для обработки файлов расположены в папке processing_scsripts. </p>
<p> В качестве примеров использования вычислительных модулей удобно использовать код скриптов из папки processing_scipts. Описание использования самих скриптов можно найти в шапках самих файлов</p>

## Незавершенные этапы обработки <a name="unfinished"> </a>
<ul>
<li> Совмещение кадров </li>
<li> Преобразование развернутой фазы к карте высот </li>
</ul>