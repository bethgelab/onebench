# Modified from the fantastic choix package

from choix.lsr import _init_lsr
import numpy as np
import scipy.linalg as spl
import warnings


def lsr_top1(n_items, data, alpha=0.1, initial_params=None):
    weights, chain = _init_lsr(n_items, alpha, initial_params)

    for winner, losers in data:
        loser_weights_sum = weights.take(losers).sum()
        total_weight = loser_weights_sum + weights[winner]

        if total_weight == 0:
            # continue
            total_weight = 1e-10

        for loser in losers:
            chain[loser, winner] += 1 / (weights[winner] + weights[loser])

    chain -= np.diag(chain.sum(axis=1))

    return log_transform(statdist(chain))



def statdist(generator):
    generator = np.asarray(generator)
    n = generator.shape[0]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lu, piv = spl.lu_factor(generator.T, check_finite=False)

    left = lu[:-1, :-1]
    right = -lu[:-1, -1]

    try:
        res = spl.solve_triangular(left, right, check_finite=False)
    except Exception as e:
        raise ValueError("Stationary distribution could not be computed. "
                         "Perhaps the Markov chain has more than one absorbing class?") from e

    res = np.append(res, 1.0)
    return (n / res.sum()) * res


def log_transform(weights):
    params = np.log(weights)
    return params - params.mean()
