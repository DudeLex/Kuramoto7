import numpy as np  # для мат функций и тд (np
from scipy.integrate import odeint  # Решения интегральных и обычных дифференциальных уравнений
import matplotlib.pyplot as plt  # отрисовки всевозможных видов графиков
from random import randint


def kur2(phi, t, K1, K2, w1, w2):  # решение наших дифуров функция с ними
    phi1, phi2 = phi
    dphi1 = w1 + (K1 / N) * (np.sin(phi2 - phi1))
    dphi2 = w2 + (K2 / N) * (np.sin(phi1 - phi2))

    return [dphi1, dphi2]


tfin = 5000 # размерность по времени
K1 = 1.06  # коэфициент связи для того который хочет догнать
K2 = -1  # коэф которого не хочет чтоб его догнали
N = 2  # количество оссициляторов
R = 10  # радиус нашего круга (вообще неинтересно) условный нашего стадиона
phi0 = np.random.rand(2) * 2 * np.pi  # откуда начинают двигаться наши бегуны
w1 = 12 / (2 * np.pi * R)  # скорость первого
w2 = 10 / (2 * np.pi * R)  # скорость второго

T = tfin  #

t = np.arange(0, tfin, 0.1)  # массив времени в каждой точке чтобы смотреть более гладкая функция получается
sol = odeint(kur2, phi0, t, args=(K1, K2, w1, w2))  # решение нащего дифура получаем

wi1 = np.diff(sol[:, 0]) / np.diff(t)  # дифференцируем наш sol для того чтобы найти омеги
wi2 = np.diff(sol[:, 1]) / np.diff(t)

plt.plot(t[:-1], wi1, label='wi1')
plt.plot(t[:-1], wi2, label='wi2')

plt.xlabel('t')
plt.ylabel('frequency')
plt.legend()
plt.show()
