from RatingParser import ratingParser as rp
from Cluster import *
from time import time
from matrix_factorization_soln import MatrixFactorizationRec
from numpy.random import rand
import numpy as np
import logging

# construct the logger
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
logFormatter = logging.Formatter("%(asctime)s [%(threadName)s] [%(levelname)s]  %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

def main():
    movies_data_fname = "../data/movies.dat"
    users_data_fname = "../data/users.dat"
    ratings_data_fname = "../data/training_ratings_for_kaggle_comp.csv"
    submission_data_fname = "../data/test_submission.csv"
    #ratings_data_contents, ratings_mat = rp.get_ratings_data(ratings_data_fname)
    #movies, all_categories = rp.get_movies_data(movies_data_fname)
    #users = rp.get_users_data(users_data_fname)

    # matrix factorization model
    #mf_model = MatrixFactorizationRec()
    #mf_model.fit(ratings_mat, report_run_time=True)
    #complete_ratings_mat = mf_model.pred_all_users(report_run_time=True)
    #similarity = create_similarity(complete_ratings_mat, report_run_time=True)
    #rp.store_similarity_data(similarity, "../data/test_similarity.dat", report_run_time=True)

    # # neighbour model
    # #ratings_mat = np.matrix(np.zeros((100, 100)))
    # neighbour_m = NeighbourModel(ratings_mat)
    # neighbour_m.build_ratings_mat()

    # cluster
    similarity = create_test_similarity(10000)
    clusters = normalized_clutering_ng(similarity, 100)
    print(clusters)

def create_similarity(ratings_mat, report_run_time=False):
    start_time = time()
    util_mat = ratings_mat
    row, col = util_mat.shape
    similarity = np.matrix(np.zeros((row, row)))
    from numpy import linalg as LA

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
    if report_run_time:
        print("Execution time: %f seconds" % (time() - start_time))
    return similarity

def create_test_similarity(n):
    similarity = np.matrix(np.zeros((n, n)))
    for i in range(n):
        for j in range(i + 1, n):
            similarity[i, j] = rand()
            similarity[j, i] = similarity[i, j]
    return similarity

if __name__ == '__main__':
    main()

