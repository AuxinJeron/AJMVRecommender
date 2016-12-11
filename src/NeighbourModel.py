from numpy import zeros, matrix
from numpy.random import rand
import numpy as np
import logging

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)

class NeighbourModel:
    def __init__(self, rating_mat, movies, users):
        self.rating_mat = rating_mat.tocsr()
        self.movies = movies
        self.movie_num = len(self.movies)
        self.users = users
        self.user_num = len(self.users)

        self.user_mat, self.movie_mat = self.uv_decomposition(self.rating_mat)
        self.built_rating_mat = np.dot(self.user_mat, self.movie_mat)

        logger.info(self.user_mat)
        logger.info(self.movie_mat)

        self.similarity = matrix(zeros(self.user_num, self.user_num))

    # UV-decomposition algorithm
    def uv_decomposition(self, util_mat, n_features=8, lr=0.001, reg=0.1):
        row, col = util_mat.shape
        U = matrix(
            rand(n_features, row).reshape([row, n_features]))
        V = matrix(
            rand(n_features, col).reshape([n_features, col]))
        for c in range(1000):
            for i in range(row):
                # only deal with the known element
                for j in util_mat.indices:
                    err = util_mat[i, j] - np.dot(U[i], V[:row, j])
                    U[i] = U[i] + lr * (err * U[i] - reg * U[i])
                    V[:row, j] = V[:row, j] + (V[:row, j] * err - V[:row, j] * reg) * lr
        return U, V

    

