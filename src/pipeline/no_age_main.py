'''
Created on May 16, 2025

@author: jarpy
'''

from preprocessing.data_preprocessing import preprocess_data_knn
from training.train_knn import train_knn_model



def no_age_main():
    """ """
    train_knn_model(True)

if __name__ == '__main__':
    no_age_main()