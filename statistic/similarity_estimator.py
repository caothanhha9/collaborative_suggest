import numpy as np
from itertools import imap
# from scipy.stats.stats import pearsonr


def item_cosine_estimate(item_vector_a, item_vector_b):
    """
    :param item_vector_a:  e.g. [0, 1, 0]
    :param item_vector_b: e.g. [1, 0, 1]
    NOTE: if item vector is in form of boolean => .astype(int)
    :return: value between 0.0 and 1.0 if OK -1.0 if NOT
    """
    dot_product = np.dot(item_vector_a, item_vector_b)
    item_vector_a_mod = np.linalg.norm(item_vector_a)
    item_vector_b_mod = np.linalg.norm(item_vector_b)
    cosine_similarity = dot_product / (item_vector_a_mod * item_vector_b_mod)
    return cosine_similarity


def pearsonr(x, y):
    # Assume len(x) == len(y)
    n = len(x)
    sum_x = float(sum(x))
    sum_y = float(sum(y))
    sum_x_sq = sum(map(lambda x: pow(x, 2), x))
    sum_y_sq = sum(map(lambda x: pow(x, 2), y))
    psum = sum(imap(lambda x, y: x * y, x, y))
    num = psum - (sum_x * sum_y/n)
    den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
    if den == 0:
        return 0
    return num / den


def average(x):
    n = len(x)
    sum_x = float(sum(x))
    if n == 0:
        return 0
    return sum_x / n
