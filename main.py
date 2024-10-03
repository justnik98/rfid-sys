import random

import pandas as pd
import numpy as np


def round_gen(L: int, N: int):
    """
    Генерирует один фрейм
    :param L: число окон в генерируемом фрейме
    :param N: число активных меток
    :return: число "успехов", "пусто" и "конфликтов" в фрейме
    """
    r = np.zeros(L)
    for i in range(0, N):
        rand = random.randint(0, L - 1)
        r[rand] += 1
    s = sum(i == 1 for i in r)
    e = sum(i == 0 for i in r)
    c = sum(i > 1 for i in r)
    return s, e, c


def generate_data(num: int, N_max: int, L_max: int):
    data_x = []
    data_y = []
    for i in range(0, num):
        for L in range(1, L_max + 1):
            for N in range(1, N_max + 1):
                s, e, c = round_gen(L, N)
                x = [s, e, c, L]
                data_x.append(x)
                data_y.append(N)
    df_x = pd.DataFrame(data_x)
    df_y = pd.DataFrame(data_y)
    df_x.to_csv('./data_x', header=False, index=False)
    df_y.to_csv('./data_y', header=False, index=False)


generate_data(1000, 10, 1024)
