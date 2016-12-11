from numpy import zeros, matrix
from numpy.random import rand
from numpy import linalg as LA
from scipy import sparse
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)

class NeighbourModel:
    def __init__(self, rating_mat=None, movies=None, users=None):
        self.rating_mat = rating_mat
        if isinstance(self.rating_mat, sparse.lil_matrix):
            self.rating_mat = self.rating_mat.tocsr()
        self.complete_ratings_mat = rating_mat
        self.movies = movies
        if self.movies:
            self.movie_num = len(self.movies)
        self.users = users
        if self.users:
            self.user_num = len(self.users)
        self.user_mat = None
        self.movie_mat = None
        self.similarity = None

    def build_ratings_mat(self):
        self.user_mat, self.movie_mat = self.uv_decomposition(self.rating_mat)
        self.complete_ratings_mat = np.dot(self.user_mat, self.movie_mat)

    def store_ratings_mat(self, output_fname):
        predictions_mat = self.complete_ratings_mat
        user_n, movie_n = predictions_mat.shape
        user_col = np.zeros(user_n * movie_n, dtype=int)
        movie_col = np.zeros(user_n * movie_n, dtype=int)
        for i in range(user_n):
            for j in range(movie_n):
                user_col[movie_n * i + j] = int(i + 1)
                movie_col[movie_n * i + j] = int(j + 1)
        d = {"user": user_col, "movie": movie_col}
        df = pd.DataFrame(data=d)
        df["rating"] = df.apply(lambda x: predictions_mat[x["user"] - 1, x["movie"] - 1], axis=1)
        df["id"] = df.apply(lambda x: str(int(x["user"])) + "_" + str(int(x["movie"])), axis=1)
        df.drop(["movie"], axis=1, inplace=True)
        df.to_csv(output_fname, index=False)

    # UV-decomposition algorithm
    def uv_decomposition(self, util_mat, n_features=8, lr=0.001, reg=0.1):
        row, col = util_mat.shape
        U = matrix(
            rand(n_features, row).reshape([row, n_features]))
        V = matrix(
            rand(n_features, col).reshape([n_features, col]))
        for c in range(1):
            for i in range(row):
                # only deal with the known element
                for j in util_mat.indices:
                    err = util_mat[i, j] - np.dot(U[i], V[:row, j])
                    U[i] = U[i] + lr * (err * U[i] - reg * U[i])
                    V[:row, j] = V[:row, j] + (V[:row, j] * err - V[:row, j] * reg) * lr
        return U, V

    # build similarity matrix
    def build_similarity(self):
        util_mat = self.complete_ratings_mat
        row, col = util_mat.shape
        similarity = matrix(zeros((row, row)))

        for i in range(row):
            for j in range(i + 1, row):
                v_i = util_mat[i].getA1()
                print(v_i)
                mean_i = np.mean(v_i)
                v_i = v_i - mean_i
                v_j = util_mat[j].getA1()
                mean_j = np.mean(v_j)
                v_j = v_j - mean_j
                similarity[i, j] = np.dot(v_i, np.transpose(v_j)) / LA.norm(v_i) / LA.norm(v_j)
                similarity[j, i] = similarity[i, j]

        self.similarity = similarity

