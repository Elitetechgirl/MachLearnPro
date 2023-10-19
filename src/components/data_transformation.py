### 2. For feature engeneering, data cleaning, and to convert categorial features into numerical features.
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer #use to create the pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import customException
from src.logger import logging
import os 

from src.utils import save_object

@dataclass
class DataTransformationConfig:  #create all input that are required for the data transformation
    preprocessor_obj_file_path=os.path.join('artifacts', "proprocessor.pkl") #to save the model in a path.

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):  #to create all my pikle files,which will be responsibleto converting categorical features into numerical.
        ''' This function is responsible for data transformation
        '''
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=["gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course"]


            num_pipeline=Pipeline(                                 # pipeline for numerical values.   
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  #impute missing val with the mean
                    ("scaler", StandardScaler())                    # is doing the standard scalling
                ]
            )
            

            cat_pipeline=Pipeline(                                       #pipeline for categorial variable
                steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),                       #we will use onehotencoder
                ("scaler",StandardScaler(with_mean=False))                             #we will use standadscaler
                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns), # we apply numerical pipeline to numerical columns
                    ("cat_pipelines",cat_pipeline,categorical_columns) # Same here

                ]
            )

            return preprocessor


        except Exception as e:
            raise customException(e, sys)
            

# let start our data transformation technique

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object() 

            target_column_name="math_score"

            numerical_columns=["writing_score","reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            

            train_arr=np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
                ]
            test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # paths for saving
            logging.info(f"Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
           raise customException(e, sys)

    #then we will use utils
    
