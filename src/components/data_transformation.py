import sys
from dataclasses import dataclass # to create a class variable and not any functionalities

import numpy as np 
from numpy import array
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2

from scipy.sparse import csr_matrix

from src.exception import CustomException
from src.logger import logging


import os
from src.utils import save_object


# here we will do feature engineering; handling: missing values, outlyers; feature scaling, handling catagorical & numerical features 
# input= train & test data path, output = transformed data, pickle files
@dataclass # to create a class variable and not any functionalities
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl') # give path of pickle files

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def target_encode(self, df):
        try:
            logging.info('target encoding initiated')
            target_map= {
                "p": 0, 
                "e": 1
                }
            df= df['classs'].map(target_map)
            return df
        except Exception as e:
            logging.info("Error in target encoding")
            raise CustomException(e,sys)


    def get_data_transformation_object(self):
        try:

            categorical_cols = ['cap_surface', 'bruises', 'gill_spacing', 'gill_size', 'gill_color',
            'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
            'ring_type', 'spore_print_color', 'population', 'habitat'] # keeping only the important columns for our model


            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehotencoder',OneHotEncoder(handle_unknown='ignore', drop= 'first', sparse_output= True))
                # no need to standardize after get dummies
                ]

            )
            # combine target and catagorical pipeline
            preprocessor=ColumnTransformer([
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ], remainder ='passthrough', n_jobs=-1)
            return preprocessor

        except Exception as e:
            raise e
        
    def initaite_data_transformation(self,X_train_path,X_test_path, y_train_path,y_test_path):
        try:
            # Reading train and test data
            X_train_df = pd.read_csv(X_train_path)
            X_test_df = pd.read_csv(X_test_path)
            y_train_df = pd.read_csv(y_train_path)
            y_test_df = pd.read_csv(y_test_path)

            logging.info('Read train and test data completed')
            logging.info(f'X-Train Dataframe Head : \n{X_train_df.head().to_string()}')
            logging.info(f'X-Test Dataframe Head  : \n{X_test_df.head().to_string()}')
            logging.info(f'y-Train Dataframe Head : \n{y_train_df.head().to_string()}')
            logging.info(f'y-Test Dataframe Head  : \n{y_test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            # feature engineering

            ## Trnasformating X using preprocessor obj
            categorical_cols = ['cap_surface', 'bruises', 'gill_spacing', 'gill_size', 'gill_color',
            'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
            'ring_type', 'spore_print_color', 'population', 'habitat'] # keeping only the important columns for our model

            X_train_arr=preprocessing_obj.fit_transform(X_train_df[categorical_cols])
            X_test_arr=preprocessing_obj.transform(X_test_df[categorical_cols])

            # Transforming y using target encode:
            y_train_arr=self.target_encode(y_train_df)
            y_test_arr=self.target_encode(y_test_df)

            # saving the pickle files
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                X_train_arr,
                X_test_arr,
                y_train_arr,
                y_test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
        


