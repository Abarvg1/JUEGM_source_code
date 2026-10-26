import mpmath as mp
from math import factorial
from collections import Counter


# ============================================================
# Part 1. Partitions and symmetric-group data
# ============================================================

def partition_size(lam):
    return sum(lam)


def partition_length(lam):
    return len(lam)


def conjugate_partition(lam):
    """Conjugate partition lambda'."""
    if not lam:
        return []

    return [
        sum(1 for x in lam if x >= j)
        for j in range(1, lam[0] + 1)
    ]


def hook_lengths(lam):
    """
    Hook lengths h_{ij} for all boxes (i,j) in lambda.
    Indices i,j are 1-based.
    """
    lam_conj = conjugate_partition(lam)
    hooks = []

    for i, row_length in enumerate(lam, start=1):
        for j in range(1, row_length + 1):
            h = (
                row_length - j
                + lam_conj[j - 1] - i
                + 1
            )
            hooks.append(h)

    return hooks


# ============================================================
# Character of the symmetric group chi^lambda(mu)
# ============================================================
#
# We calculate chi^lambda(mu) using the Murnaghan-Nakayama rule.
#
# For the moderate sizes normally encountered in these
# calculations this is sufficient.
# ============================================================

def cells_of_partition(lam):
    """Set of cells (i,j) belonging to lambda."""
    return {
        (i, j)
        for i, row_len in enumerate(lam, start=1)
        for j in range(1, row_len + 1)
    }


def normalize_partition(cells):
    """
    Convert a set of cells into a partition.
    Return None if cells do not form a Young diagram.
    """
    if not cells:
        return ()

    max_i = max(i for i, j in cells)

    lengths = []
    for i in range(1, max_i + 1):
        row = [j for ii, j in cells if ii == i]

        if not row:
            lengths.append(0)
        else:
            if set(row) != set(range(1, max(row) + 1)):
                return None
            lengths.append(len(row))

    # Remove zero rows at the end
    while lengths and lengths[-1] == 0:
        lengths.pop()

    # Rows must be non-increasing
    if any(lengths[i] < lengths[i + 1]
           for i in range(len(lengths) - 1)):
        return None

    return tuple(lengths)


def is_border_strip(removed):
    """
    Check whether 'removed' is a border strip (rim hook).

    Conditions:
      - connected by edges
      - contains no 2x2 block
    """
    if not removed:
        return False

    # Edge connectivity
    remaining = {next(iter(removed))}
    changed = True

    while changed:
        changed = False
        for cell in removed - remaining:
            i, j = cell
            if any((i + di, j + dj) in remaining
                   for di, dj in [(1, 0), (-1, 0),
                                  (0, 1), (0, -1)]):
                remaining.add(cell)
                changed = True

    if remaining != removed:
        return False

    # No 2x2 block
    for i, j in removed:
        if ((i + 1, j) in removed and
            (i, j + 1) in removed and
            (i + 1, j + 1) in removed):
            return False

    return True


def removable_border_strips(lam, k):
    """
    Find all partitions nu obtained from lambda by removing
    a border strip of k cells.
    """
    lam_cells = cells_of_partition(lam)
    n = len(lam_cells)

    if k > n:
        return []

    result = {}

    # For small/medium n this exhaustive enumeration is OK.
    # We choose k cells to remove.
    from itertools import combinations

    for removed_tuple in combinations(lam_cells, k):
        removed = set(removed_tuple)
        remaining = lam_cells - removed

        nu = normalize_partition(remaining)

        if nu is None:
            continue

        if not is_border_strip(removed):
            continue

        rows = {i for i, j in removed}
        height = len(rows) - 1

        result[nu] = height

    return list(result.items())


from functools import lru_cache


@lru_cache(None)
def character_symmetric_group(lam, mu):
    """
    Murnaghan-Nakayama rule for chi^lambda(mu).

    lam and mu must be tuples.
    """
    lam = tuple(lam)
    mu = tuple(mu)

    if not mu:
        return 1 if sum(lam) == 0 else 0

    if sum(lam) != sum(mu):
        return 0

    # Remove the first cycle length.
    k = mu[0]
    rest = mu[1:]

    total = 0

    for nu, height in removable_border_strips(lam, k):
        total += (-1) ** height * character_symmetric_group(
            nu, rest
        )

    return total


# ============================================================
# Partitions of n
# ============================================================

def integer_partitions(n, max_part=None):
    """Generate partitions of n in decreasing order."""
    if n == 0:
        yield ()
        return

    if max_part is None or max_part > n:
        max_part = n

    for first in range(max_part, 0, -1):
        for rest in integer_partitions(n - first, first):
            yield (first,) + rest


# ============================================================
# z_mu
# ============================================================

def z_mu(mu):
    """
    z_mu = product_i i^(m_i) m_i!
    """
    counts = Counter(mu)
    z = 1

    for k, m in counts.items():
        z *= k ** m
        z *= factorial(m)

    return z


# ============================================================
# c_{lambda N}
# ============================================================

def c_lambda_N(lam, N, cA, cB, q, p, mp_dps=50):
    """
    Calculate c_{lambda N} from the formula in the uploaded file.
    """

    mp.mp.dps = mp_dps

    n = sum(lam)

    if n == 0:
        return mp.mpf(1)

    eps = mp.mpf(1) / N

    # Product over boxes (i,j) in lambda
    product = mp.mpf(1)

    for i, row_length in enumerate(lam, start=1):
        for j in range(1, row_length + 1):
            content = j - i

            numerator = 1 + eps * content

            denominator = (
                1
                + eps * content / (cA + cB)
            )

            product *= numerator / denominator

    # Sum over mu partitions of n
    summation = mp.mpf(0)

    for mu in integer_partitions(n):

        ell = len(mu)

        chi = character_symmetric_group(
            tuple(lam),
            tuple(mu)
        )

        if chi == 0:
            continue

        term = mp.mpf(chi) / z_mu(mu)

        term *= eps ** (n - ell)

        term /= (cA + cB) ** n

        for part in mu:
            term *= (
                q ** part * cA
                + p ** part * cB
            )

        summation += term

    r = product * summation

    return r * N ** n


# ============================================================
# Jacobi moments
# ============================================================

def jacobi_moment(k, p, q, alpha, beta):
    """
    M_k = integral_p^q x^k (q-x)^beta (x-p)^alpha dx

    Calculated analytically using a finite sum of beta functions.
    """

    if k < 0:
        raise ValueError("Moment index k must be non-negative.")

    if q <= p:
        raise ValueError("Require q > p.")

    if alpha <= -1 or beta <= -1:
        raise ValueError("Require alpha > -1 and beta > -1.")

    L = q - p

    result = mp.mpf(0)

    for r in range(k + 1):
        result += (
            mp.binomial(k, r)
            * p ** (k - r)
            * L ** r
            * mp.beta(
                alpha + r + 1,
                beta + 1
            )
        )

    return L ** (alpha + beta + 1) * result


# ============================================================
# Integral I
# ============================================================

def integral_I(lam, N, p, q, alpha, beta, mp_dps=50):
    """
    Calculate

        I = integral s_lambda(x) Delta(x)^2
            prod_a (q-x_a)^beta (x_a-p)^alpha dx

    using Andreief's identity.
    """

    mp.mp.dps = mp_dps

    if len(lam) > N:
        raise ValueError(
            "lambda must have at most N parts."
        )

    # Pad lambda by zeroes
    lam_full = tuple(lam) + (0,) * (N - len(lam))

    # Matrix of moments
    M = mp.matrix(N, N)

    for i in range(1, N + 1):
        for j in range(1, N + 1):

            k = (
                lam_full[i - 1]
                + 2 * N
                - i
                - j
            )

            M[i - 1, j - 1] = jacobi_moment(
                k, p, q, alpha, beta
            )

    return mp.factorial(N) * mp.det(M)


# ============================================================
# Complete calculation
# ============================================================

def calculate_all(
    lam,
    N,
    cA,
    cB,
    p,
    q,
    alpha,
    beta,
    mp_dps=50
):

    mp.mp.dps = mp_dps

    # --------------------------------------------------------
    # Normalized integral
    # --------------------------------------------------------

    I_norm = normalized_integral(
        lam=lam,
        N=N,
        p=p,
        q=q,
        alpha=alpha,
        beta=beta,
        mp_dps=mp_dps
    )

    # --------------------------------------------------------
    # c_lambda_N
    #
    # NO normalization!
    # --------------------------------------------------------

    c = c_lambda_N(
        lam=lam,
        N=N,
        cA=cA,
        cB=cB,
        q=q,
        p=p,
        mp_dps=mp_dps
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    if abs(c) < mp.mpf("1e-100"):
        ratio = mp.inf
    else:
        ratio = I_norm / c

    difference = I_norm - c

    return {
        "I_normalized": I_norm,
        "c_lambda_N": c,
        "I_normalized / c_lambda_N": ratio,
        "I_normalized - c_lambda_N": difference,
    }


def normalized_integral(
    lam,
    N,
    p,
    q,
    alpha,
    beta,
    mp_dps=50
):
    """
    Normalized integral

        I_norm(lambda) = I(lambda) / I(lambda = 0)
    """

    mp.mp.dps = mp_dps

    I_lambda = integral_I(
        lam=lam,
        N=N,
        p=p,
        q=q,
        alpha=alpha,
        beta=beta,
        mp_dps=mp_dps
    )

    I_zero = integral_I(
        lam=(),
        N=N,
        p=p,
        q=q,
        alpha=alpha,
        beta=beta,
        mp_dps=mp_dps
    )

    if abs(I_zero) < mp.mpf("1e-100"):
        raise ZeroDivisionError(
            "I(lambda=0) is zero."
        )

    return I_lambda / I_zero

# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    # Example input
    lam = (6,4,2,1)

    N = 11

    cA = mp.mpf("2.2")
    cB = mp.mpf("1.2")

    p = mp.mpf("-1.4")
    q = mp.mpf("3.2")

    alpha = (cA - 1)*N
    beta = (cB - 1)*N

    result = calculate_all(
        lam=lam,
        N=N,
        cA=cA,
        cB=cB,
        p=p,
        q=q,
        alpha=alpha,
        beta=beta,
        mp_dps=80
    )

    for name, value in result.items():
        print(f"{name} =")
        print(mp.nstr(value, 40))
        print()