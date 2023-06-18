import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass # when you just need to create a class variable and not any functionalities

from src.components.data_transformation import DataTransformation
from src.utils import connect_database



# input= clean data path, output = train and test data path
## Intitialize the Data Ingetion Configuration

@dataclass
class DataIngestionconfig:      # put all the paths here, directory paths are in the form of string
    X_train_data_path:str=os.path.join('artifacts','X_train.csv')
    X_test_data_path:str=os.path.join('artifacts','X_test.csv')
    y_train_data_path:str=os.path.join('artifacts','y_train.csv')
    y_test_data_path:str=os.path.join('artifacts','y_test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

## create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()     # just the above created class for specifying the paths

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            df = pd.DataFrame([d for d in connect_database()]) # read data from local, mongodb, sql (write all these generic codes in utils)
            df = df.iloc[:-1 , :] # # By using iloc[] to select all rows except the last row
            df["stalk_root"]= df["stalk_root"].replace('?', np.nan)
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True) # make directory, if already exists don't worry
            df.to_csv(self.ingestion_config.raw_data_path,index=False) # this file will be created
            logging.info('Train test split')
            X=df.drop(labels=['classs'], axis=1)
            y=df['classs']
            X_train, X_test, y_train, y_test=train_test_split(X, y,test_size=0.20,random_state=42, stratify= y)
            # EDA is done before hand in notebooks, this is test train split

            X_train.to_csv(self.ingestion_config.X_train_data_path,index=False,header=True)     # create train test data files
            X_test.to_csv(self.ingestion_config.X_test_data_path,index=False,header=True)
            y_train.to_csv(self.ingestion_config.y_train_data_path,index=False,header=True)
            y_test.to_csv(self.ingestion_config.y_test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(     # return train data & test data path
                self.ingestion_config.X_train_data_path,
                self.ingestion_config.X_test_data_path,
                self.ingestion_config.y_train_data_path,
                self.ingestion_config.y_test_data_path
            )
  
            
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)


'''
if __name__=='__main__':
    obj=DataIngestion()
    X_train_data_path, X_test_data_path, y_train_data_path, y_test_data_path=obj.initiate_data_ingestion()
    data_transformation= DataTransformation()
    X_train_arr, X_test_arr, y_train_arr, y_test_arr,_= data_transformation.initaite_data_transformation(X_train_data_path, X_test_data_path, y_train_data_path, y_test_data_path)
'''



