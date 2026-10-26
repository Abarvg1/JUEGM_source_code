
# SageMath

from sage.all import *
from functools import lru_cache
from collections import Counter


# Character chi^lambda(mu) via the Frobenius formula
@lru_cache(None)
def chi(lam, mu):
    if sum(lam) != sum(mu):
        return 0
    if not mu:
        return int(not lam)

    Sym = SymmetricFunctions(QQ)
    s, p = Sym.schur(), Sym.power()
    return s(p[Partition(mu)]).coefficient(Partition(lam))


def z(mu):
    return prod(k**m * factorial(m)
                for k, m in Counter(mu).items())


def c_lambda_N(lam, N, cA, cB, p, q):
    lam = Partition(lam)
    n = lam.size()
    if n == 0:
        return 1

    eps = 1 / N
    C = cA + cB

    prefactor = prod(
        (1 + eps*(j-i)) / (1 + eps*(j-i)/C)
        for i, row in enumerate(lam, 1)
        for j in range(1, row + 1)
    )

    total = 0
    for mu in Partitions(n):
        character = chi(tuple(lam), tuple(mu))
        if character:
            total += (
                character / z(mu)
                * eps**(n-len(mu)) / C**n
                * prod(q**k*cA + p**k*cB for k in mu)
            )

    return N**n * prefactor * total


def moment(k, p, q, alpha, beta):
    if k < 0 or q <= p or alpha <= -1 or beta <= -1:
        raise ValueError("Invalid moment parameters")

    R = parent(p)
    L = q - p

    return L**(alpha+beta+1) * sum(
        binomial(k, r) * p**(k-r) * L**r
        * R(alpha+r+1).gamma() * R(beta+1).gamma()
        / R(alpha+beta+r+2).gamma()
        for r in range(k+1)
    )


def integral_I(lam, N, p, q, alpha, beta):
    lam = Partition(lam)
    if len(lam) > N:
        raise ValueError("lambda must have at most N parts")

    lam = list(lam) + [0]*(N-len(lam))
    M = matrix(parent(p), N, N, lambda i, j:
        moment(lam[i] + 2*N-i-j-2, p, q, alpha, beta)
    )
    return factorial(N) * M.det()


def calculate_all(lam, N, cA, cB, p, q, alpha, beta):
    I0 = integral_I([], N, p, q, alpha, beta)
    if I0 == 0:
        raise ZeroDivisionError("I(lambda=0) is zero")

    I_norm = integral_I(lam, N, p, q, alpha, beta) / I0
    c = c_lambda_N(lam, N, cA, cB, p, q)

    return {
        "I_normalized": I_norm,
        "c_lambda_N": c,
        "I_normalized / c_lambda_N": I_norm/c if c else Infinity,
        "I_normalized - c_lambda_N": I_norm-c,
    }


# Example
R = RealField(800)
lam = [6, 4, 2, 1]
N = 11

cA, cB = R("2.2"), R("1.2")
p, q = R("-1.4"), R("3.2")
alpha, beta = (cA-1)*N, (cB-1)*N

I0 = integral_I([], N, p, q, alpha, beta)
print("I0 =", I0)
print("I_norm(empty) =", integral_I([], N, p, q, alpha, beta)/I0)

for name, value in calculate_all(
        lam, N, cA, cB, p, q, alpha, beta).items():
    print(f"{name} =\n{value}\n")
