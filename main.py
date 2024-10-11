from modeling import *
#import gmpy2
#from gmpy2 import mpfr, mpz, add, div, factorial

#gmpy2.get_context().precision=2000
# def ns_pr(k, n, V):
#         k = mpz(k)
#         n = mpz(n)
#         V = mpz(V)
#         sum = 0.0
#         res1 = mpfr(mpfr(-1) ** k * factorial(V)) *mpfr(factorial(n)) / mpfr(((V ** n) * factorial(k)))
#         for j in range(k, min(V, n) + 1):
#             sum += ((-1) ** j * (V - j) ** (n - j))/ (math.factorial(j - k) * math.factorial(V - j) * math.factorial(n - j))
#         res = res1 * sum
#         return res
#
#
# def lower_bound(N):
#     res = np.zeros(N + 1)
#     res[1] = 1
#     for i in range(2, N + 1):
#         print (i)
#         for j in range(1, i + 1):
#             res[i] += (i + res[i - j]) * ns_pr(j, i, i)
#         res[i] += i * ns_pr(0, i, i)
#         res[i] /= (1 - ns_pr(0, i, i))
#     return res

def lower_bound(N):
    res = np.zeros(N + 1)
    res[1] = 1
    for n in range(2, N + 1):
        print (n)
        res [n] = n*np.e
    return res
def plot_lb(N):
    res = lower_bound(N)
    plt.title("Average delay")
    plt.legend(["Lower bound (theory)"])
    plt.xlabel("Number of tags")
    plt.ylabel("Average delay, slots")
    plt.plot(res)
    plt.grid()


def modeling(N):
    xf = open('./data_x', 'w')
    mlf = open('./data_ml', 'w')
    qal = open('./data_qa', 'w')
    lbf = open('./data_lb', 'w')
    Num = 100
    q_alg = []  # np.zeros(N + 1)
    ml_alg = []  # np.zeros(N + 1)
    lb = []
    clfs = createClfs(N, True)
    x = []
    for n in range(0, N + 1, 50):
        print(n)
        x.append(n)
        sum1 = 0
        sum2 = 0
        sum3 = 0
        for i in range(0, Num):
            sum1 += modeling_MLalg(n, clfs)
            sum2 += modeling_Qalg(n)
            sum3 += modeling_lb(n)

        ml_alg.append(sum1 / Num)
        q_alg.append(sum2 / Num)
        lb.append(sum3 / Num)
        xf.write(f'{str(x[-1])}\n')
        mlf.write(f'{str(ml_alg[-1])}\n')
        qal.write(f'{str(ml_alg[-1])}\n')
        lbf.write(f'{str(lb[-1])}\n')

    plot_lb(N)
    plt.plot(x, ml_alg)

    plt.plot(x, q_alg)
    plt.legend(["lower_bound" ,"ML-alg delay (modeling)", "Q-alg delay (modeling)"])
    plt.title("Average delay")
    plt.xlabel("Number of tags")
    plt.ylabel("Average delay, slots")
    plt.show()


modeling(1000)
