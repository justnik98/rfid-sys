import random
import math

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVR


def m_round(N, V):
    r = np.zeros(V)
    for i in range(0, N):
        rand = random.randint(0, V - 1)
        r[rand] += 1
    s = sum(i == 1 for i in r)
    e = sum(i == 0 for i in r)
    c = sum(i > 1 for i in r)
    return s, e, c


def m_round_(N, V):
    r = np.zeros(V)
    for i in range(0, N):
        rand = random.randint(0, V - 1)
        r[rand] += 1
    return r


def modeling_lb(N):
    n = N
    delay = 0
    while n != 0:
        delay += n
        [k, x, y] = m_round(n, n)
        n -= k
    return delay


def modeling_Qalg(N):
    n = N
    q = 4.0
    delta = 0.1
    delay = 0

    while n != 0:
        if (q >= 1):
            delta = min([max([0.8 / round(q), 0.1]), 0.5])
        V = 2 ** round(q)
        new_q = q
        # delay += V
        r = m_round_(n, V)
        for i in range(0, len(r)):
            if r[i] == 1:
                n -= 1

            elif r[i] == 0:
                new_q = max(0, new_q - delta)
                if abs(round(new_q) - round(q)) == 1:
                    delay += 3
                    break
            else:
                new_q = min(15, new_q + delta)
                if abs(round(new_q) - round(q)) == 1:
                    delay +=  3
                    break
            delay += 1
        q = new_q
    return delay


def modeling_MLalg(N, clfs):
    n = N
    q = 4
    V = 2 ** q
    delay = 0
    while n != 0:
        delay += V + 2
        k, e, c = m_round(n, V)
        n -= k
        tmp = clfs[q].predict(np.array([k, e, c]).reshape(1, -1))
        if (tmp >= 1):
            q = round(math.log2(tmp))
        else:
            q = 1
        q = min(max(0, q), 13)
        V = 2 ** q
    return delay


clfs = []

def gen_data(Num):
    data = [[]] * 13
    y = [[]] * 13
    for i in range(0, Num):
        q = 4.0
        delta = 0.1
        delay = 0
        n = 1000
        while n != 0:
            if (q >= 1):
                delta = min([max([0.8 / round(q), 0.1]), 0.5])
            V = 2 ** round(q)
            new_q = q
            # delay += V
            r = m_round_(n, V)
            s = sum(i == 1 for i in r)
            e = sum(i == 0 for i in r)
            c = sum(i > 1 for i in r)
            data[0].append([s, e, c])
            y[0].append(n)
            for i in range(0, len(r)):
                if r[i] == 1:
                    n -= 1
                elif r[i] == 0:
                    new_q = max(0, new_q - delta)
                    if abs(round(new_q) - round(q)) == 1:
                        delay += i + 3
                        break
                else:
                    new_q = min(15, new_q + delta)
                    if abs(round(new_q) - round(q)) == 1:
                        delay += i + 3
                        break
            q = new_q
    return data, y


# def prediction_mod():
#     V = 16
#     N = 100
#     Num = 5000
#     data, y = gen_data(N, V, Num)
#     # clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, n_jobs= 3))
#     clf = make_pipeline(StandardScaler(), SVC(max_iter=1000, gamma='auto'))
#     clf.fit(data, y)
#     err = np.zeros(N)
#     pr = np.zeros(N)
#     for i in range(0, int(Num / N)):
#         for n in range(1, N):
#             s, e, c = m_round(n, V)
#             test = np.array([s, e, c])
#             err[n] += abs(n - clf.predict(test.reshape(1, -1)))
#             pr[n] += clf.predict(test.reshape(1, -1))
#     err /= (Num / N)
#     pr /= (Num / N)
#     plt.plot(err)
#     plt.plot(pr)
#     plt.show()
#     return clf


# prediction_mod()


def createClfs(N, is_fitted):
    if not is_fitted:
        svr = SVR()

        Num = 10
        data, y = gen_data(Num)
        for q in range(0,13):
            pipe = Pipeline(steps=[('scaler', StandardScaler()), ('clf', svr)])
            parameters = {'clf__kernel': (['rbf']), 'clf__C': [0.5, 1, 3, 5, 10], 'clf__max_iter': [1000, 500, 100, -1]}
            clf = GridSearchCV(pipe, parameters, n_jobs=-1, verbose=0)
            clfs.append(clf)
            clfs[q].fit(data[q], y[q])
            L = int(2 ** q)
            joblib.dump(clfs[q], f'./svr_model_L{L}.pkl')
    else:
        for q in range(0, 13):
            L = int(2 ** q)
            clf = joblib.load(f'./svr_model_L{L}.pkl')
            clfs.append(clf)
        return clfs
