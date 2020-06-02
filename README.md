# Построение генеративной модели для симуляции отклика трекера детектора MPD на ускорителе НИКА

1. [Физический background](#physical-background)
1. [Метрики качества](#метрики-качества)
1. [Логарифмирование амплитуды](#curator)
1. [Технические ремарки]


## Physical Background

В камере детектора движется частица, сталкиваясь с молекулами газа. 
Столкновения приводят к ионизации молекул газа и появлению свободных электронов (первичной ионизации),
который сталкиваясь может производить еще электроны (вторичной ионизации). 
Электрон первичной ионизации и вторичной ионизации образуют ионизационный кластер, который дрейфует под воздействием электрического поля
к левой стенке камеры, подходя к анодным проволочкам.Они индуцируют заряды на катодных пэдах. Информация об амплитуде этих зарядов собирается с пэдов.

Изображние камеры детектора:
<img src="images/MPDbarrel.png" width="200"/>


![drawing](images/MPDbarrel.png){ width=50% }

## Метрики качества
Ноутбуки с подсчетами метрики можно нйти в папке notebooks
1. https://github.com/prutianova-nastia/PhysGAN/blob/master/notebooks/Parametrized_model_results.ipynb


## Про логарифмирование амплитуды



# GANS
Sourse of inspiration:  https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
###To run:
* source ./venv/bin/activate - асtivation command
* python run.py
* python load_model.py (to plot resultes)
