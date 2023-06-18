
import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__=='__main__':
    obj=DataIngestion()
    X_train_data_path, X_test_data_path, y_train_data_path, y_test_data_path=obj.initiate_data_ingestion()
    data_transformation= DataTransformation()
    X_train_arr, X_test_arr, y_train_arr, y_test_arr,_= data_transformation.initaite_data_transformation(X_train_data_path, X_test_data_path, y_train_data_path, y_test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(X_train_arr, X_test_arr, y_train_arr, y_test_arr)




