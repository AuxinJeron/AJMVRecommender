from RatingParser import ratingParser as rp
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
    #ratings_data_contents, ratings_mat = rp.get_ratings_data(ratings_data_fname)
    movies, all_categories = rp.get_movies_data(movies_data_fname)
    users = rp.get_users_data(users_data_fname)

if __name__ == '__main__':
    main()
