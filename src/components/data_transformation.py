import os 
import sys 

import numpy as np 
import pandas as pd 
from dataclasses import dataclass 

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder , LabelEncoder , StandardScaler ,LabelEncoder  

from src.exception import CustomException
from src.logger import logging  
from src.utils import save_object



@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join("artifacts" ,"proprocessor.pkl") 
    
    
class DataTransformation :
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() 
        
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation  opearation 
        where we handle null val  , ohe ,labelencoding, standard_scaler by using pipeline 
        '''
        
        try:
            numerical_columns = ['age', 'balance', 'duration', 'pdays', 'previous'] 
            categorical_columns = ['job', 'marital', 'education', 'housing', 'loan', 'poutcome'] 
            
            # we make pipeline 
            
            num_pipeline= Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy='median')) ,# we handle null val 
                ("scaler", StandardScaler())
                
                ] ) 
            
            
            cat_pipeline=Pipeline(
                steps= [
                    ("imputer" ,SimpleImputer(strategy="most_frequent")) ,
                    ("onehotencoder" ,OneHotEncoder()) , 
                    ("scaler" ,StandardScaler(with_mean=False))    
                ]
            ) 
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}") 
            
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline" , num_pipeline , numerical_columns),
                    ("categorical_pipelien" , cat_pipeline , categorical_columns)
                ]
            ) 
            
            
            return preprocessor 
        
        
        except Exception as e :
            raise CustomException(e,sys)
        
 
 
 
    def initiate_data_transformation(self,train_path,test_path):  
        
        # here we take train , test data and apply our pipeline mehods 
        try:
            
            train_df=pd.read_csv(train_path) 
            test_df=pd.read_csv(test_path)    
            
            logging.info("reading the data complted in data_transformation") 
            
            preprocessing_obj=self.get_data_transformer_object()
            
            target_column_name='deposit' 
            
            input_feature_train_df= train_df.drop(columns=[target_column_name],axis=1) 
            target_feature_train_df = train_df[target_column_name]    
            
            
            input_feature_test_df= test_df.drop(columns=[target_column_name],axis=1) 
            target_feature_test_df = test_df[target_column_name]    
            
            logging.info(" We Applying labelencoding to target data .")
            
            label_encoder = LabelEncoder()  
            
            target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df= label_encoder.transform(target_feature_test_df)
            
            logging.info(
                    f" We Applying preprocessing object on training dataframe and testing dataframe.") 
            
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df) 
            
            # we combine train_test data in one form 
            
            train_arr= np.c_[input_feature_train_arr , np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr , np.array(target_feature_test_df)]
            
            
            logging.info(f" We Applyed all preprocesing steps to train and test data ") 
            
            save_object(self.data_transformation_config.preprocessor_obj_file_path ,
                        obj=preprocessing_obj) 
            
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
            
        except Exception as e:
            raise CustomException(e,sys)


























# we will make preprocessor file here we do null val handling , standard scaling , and onehotencoiding or labelencodeing 
# by using pipeline 