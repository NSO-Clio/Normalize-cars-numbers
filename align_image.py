import cv2
import numpy as np


def align_image(image, coords, output_size=(512, 112)):
    """
    Выравнивает номер на картинке и удаляет перспективу
    - image: numpy.ndarray, входное изображение
    - coords: список кортежей, координаты краёв номера на изображении в любом порядке
    - output_size: кортеж, (ширина, высота) выходного изображения
    Возвращает numpy.ndarray, выходное изображение
    """

    # преобразауем координаты в массив numpy
    coords1 = np.array(coords, dtype=np.float32)
    # находим прямоугольник (с минимальной площадью) для точек
    rect = cv2.minAreaRect(coords1)
    # получаем углы прямоугольника и сортируем их
    points = cv2.boxPoints(rect)
    points = np.array(sorted(points, key=lambda x: (x[1], x[0])), dtype=np.float32)
    # вычесялем матрицу перспективного преобразования
    matrix = cv2.getPerspectiveTransform(points, np.float32([[0, 0],
                                                          [output_size[0], 0],
                                                          [output_size[0], output_size[1]],
                                                          [0, output_size[1]]]))
    # применяем перспективное преобразование
    output_image = cv2.warpPerspective(image, matrix, output_size)
    return output_image


# inputim = cv2.imread('im.jpg')
# coordsim = [(12, 34), (97, 18), (218, 26), (172, 52), (4, 81), ]
# outputim = align_image(inputim, coordsim)
# cv2.imwrite('im1.jpg', outputim)
