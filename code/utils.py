import os
import pickle

def pickle_dump(file, var):
    with open('../../data/preprocessed_data/' + file +
              '.pickle', 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(file):
    return pickle.load(open('../../data/preprocessed_data/' + file +
                            '.pickle', "rb"))

def make_dir(path):
    try: os.mkdir(path)
    except OSError as error:
        print(error)
