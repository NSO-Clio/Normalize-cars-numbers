# Normalize-russian-cars-numbers

# Выравнивание изображений номеров автомобилей

# Заказчик

![beeline-Photoroom](https://github.com/NSO-Clio/Normalize-cars-numbers/assets/124351915/55a73fe2-752d-4888-9d3b-70b01f2b50d2)

Beeline, сделанно в рамках хакатона Цифровой Прорыв сезон: Искусственный Интеллект

# Проблематика

- На сегодняшний день использование видеонаблюдения для распознавания номеров автомобилей стало обычной практикой
- Однако ключевая проблема заключается в качестве распознавания номеров, особенно в условиях неблагоприятной освещённости, различных углов обзора и размытости изображений.
- Важным этапом в этом процессе является выравнивание номеров, чтобы обеспечить точность распознавания каждого символа.

![asdfghjklimage](https://github.com/NSO-Clio/Normalize-cars-numbers/assets/124351915/3ede15a7-30e6-41f9-9320-0de84b003d30)

![image](https://github.com/NSO-Clio/Normalize-cars-numbers/assets/124351915/a1460028-cced-4cad-bdc7-ae16aa8558c5)

# Итоговый продукт

Программный моудль, который можно использовать для обращения к алгоритмувыравнивания номеров, причём можно использовать это как для одной картинки, так и для нескольких сразу.

Наш алгоритм состоит из 3 шагов
1) Сегментация номерного знака
2) Преобразование в нужный размер
3) Выравнивание по трафарету

![Screenshot_94_3](https://github.com/NSO-Clio/Normalize-cars-numbers/assets/124351915/e8836455-c033-40fe-973c-1d495bb05120)


# Stack технологий

- PyTorch
- OpenCV
- NumPy
- pandas
- pillow
- roboflow
- Docker


# Наш главный плюс 

## Уникальность алгоритма

- Наша сегментационная модель для номеров автомобилей на фото обучена на созданном и размеченном нами датасете в roboflow с использованием аугментации для большего объёма данных. Скорость работы алгоритма 20 фото в секунду. Точность модели по собственной метрике показывает относительно неплохой результат – в среднем 10%
- Также мы используем строгие математические и матричные операции через OpenCV для выравнивания номера, что сильно сокращает количество ошибок


# Практическая применимость

> Реализация нашего алгоритма по выравниванию номеров значительно повышает точность и эффективность систем видеонаблюдения, использующих распознавание номеров автомобилей.
 
> Это особенно важно для бизнеса, работающего в сферах безопасности, правоохранительных органов, контроля транспортных средств, управления парковками.


# Масштабируемость

- Наш алгоритм можно использовать для выравнивания номеров и других стран, не только России. Для этого необходимо дообучить сегментационную модель на размеченных данных номеров других стран. 

- Также наш алгоритм можно использовать для большого количества входных потоков данных, при условии увеличения вычислительных ресурсов.


# Как запустить приложение?

- как запустить проект можно посмотреть по ссылке в папке [src](src/)
- В папке [notebooks](notebooks/) указаны все Jupyter Notebook-и в которых проводился анализ и обучения модели
- Веса модели можно посмотреть по ссылке https://drive.google.com/drive/folders/1NFzOVovRzFjN6iBO0q58cqdTAWhHdaTt?usp=drive_link

# Наша команда

**Андреасян Егор**
> ML-инженер
- Почта: egorandreasyan@yandex.ru
- telegram: @EgorAndrik

**Вершинин Михаил**
> ML-инженер
- Почта: m_ver08@mail.ru
- telegram: @Radsdafar08

**Сусляков Семен**
> BackEnd-разработчик
- Почта: ssuslyakoff@gmail.com
- telegram: @ssuslyakoff

**Ротачёв Александр**
> CV-инженер
- email: rotachevaa07@gmail.com
- telegram: @developweb3
