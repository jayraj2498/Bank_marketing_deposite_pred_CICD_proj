import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV 
from dotenv import load_dotenv 
import pymysql as sql  
import pymysql 

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV 
from dotenv import load_dotenv 

from src.exception import CustomException 
from src.logger import logging





# 1 . first we read sql data  
# Reading dataset from SQl 
# to read confidential data  from .env file we download liabrary -> python-dotenv (it will load all data from  .env file) 

import os 
import sys 
from src.exception import   CustomException 
from src.logger import logging 

import pandas as pd 

from dotenv import load_dotenv 
import pymysql as sql 

load_dotenv() 

host=os.getenv("host") 
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db") 


def read_sql_data() :
    logging.info("Reading sql data method started :") 
    try:
        mydb=sql.connect(
            host=host,
            user=user ,
            password=password ,
            db=db
        )
        logging.info("Connection wrt db establish as " ,mydb) 
        df=pd.read_sql_query("select * from bank_marketing_deposit_prediction " , mydb)
        # print(df.head(3))
        return df 
        
    except Exception as e:
        raise CustomException(e,sys)
    
    
    
# we define save_obeject function to save our pkl file in directory 

def save_object(file_path,obj):
    try:
        dir_path= os.path.dirname(file_path) 
        
        os.makedirs(dir_path,exist_ok=True) 
        
        with open(file_path ,"wb") as file_obj :
            pickle.dump(obj,file_obj)
        
    
    except Exception as e:
        raise CustomException(e,sys) 
    
    
    
    
# we make the function for to train all model 

def evaluate_models(X_train , y_train , X_test , y_test , models ,params) : 
    
    try:
        report={} 
        
        for i in range(len(models)):
            model=list(models.values())[i] 
            para=params[list(models.keys())[i]]   
            
            gs=GridSearchCV(model , para ,cv=4) 
            gs.fit(X_train,y_train) 
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)  
            
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            
            train_model_score =accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test,y_test_pred) 
            
            report[list(models.keys())[i]] = test_model_score    # append acc of test_model 
            
        return report
            
            
            
    except Exception as e :
        raise CustomException(e,sys) 
    
    
    
    
# now we make function to load our pkl model in predict_pipelien.py 

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e :
        raise CustomException(e,sys)