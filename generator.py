import random
import time

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


def generate_data(num, N_max: int, q_max: int):
    start = time.time()
    data_x = []
    data_y = []
    count = 0
    q = 0
    while q != q_max:
        L = 2 ** q
        for N in range(1, N_max + 1):
            for i in range(0, num):
                s, e, c = round_gen(L, N)
                x = [s, e, c, L]
                data_x.append(x)
                data_y.append(N)
                count += 1
        q += 1

        now = time.time()
        percent = q * 100 / q_max
        time_sp = round(now - start, 3)
        remained = time_sp / q * q_max - time_sp
        print(
            f'total count {count} = {round(percent, 3)} %, elapsed time = {time_sp} sec, remained = {round(remained, 3)} sec')
    df_x = pd.DataFrame(data_x)
    df_y = pd.DataFrame(data_y)
    df_x.to_csv('./data_x.csv', header=False, index=False)
    df_y.to_csv('./data_y.csv', header=False, index=False)
    return data_x, data_y


generate_data(1000, 256, 8)
