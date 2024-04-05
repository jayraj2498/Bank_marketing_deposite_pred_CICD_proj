# here we load the model and we crete dataframe function 

import os 
import sys 
import pandas as pd 

from src.exception import CustomException 
from src.logger import logging 

from src.utils import load_object


class  predictpipeline :
    def __init__(self) :
        pass 
    
    def predict(self , features) :
        try :
            model_path = os.path.join('E:\\Bank_cicd\\artifacts\\model.pkl')
            preprocessor_path= os.path.join('E:\\Bank_cicd\\artifacts\\proprocessor.pkl') 
            
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading") 
            
            scaled_data=preprocessor.transform(features)
            pred=model.predict(scaled_data)
            
            return pred  
        
        except Exception as e :
            raise CustomException(e,sys)
        
        
        
        
class CustomData: 
    def __init__(self, 
                 age: int,
                 job: str,
                 marital: str,
                 education: str,
                 balance: int,
                 housing: str,
                 loan: str,
                 duration: int,
                 pdays: int,
                 previous: int,
                 poutcome: str
                 ):
        
        
        self.age = age
        self.job = job
        self.marital = marital
        self.education = education
        self.balance = balance
        self.housing = housing
        self.loan = loan
        self.duration = duration
        self.pdays = pdays
        self.previous = previous
        self.poutcome = poutcome
        
        
    # here we make incoming data as dataframe to train our mdel     
        
    def get_data_as_data_frame(self):
        
        try:
            custom_data_input_dict ={
            "age": [self.age],
            "job": [self.job],
            "marital": [self.marital],
            "education": [self.education],
            "balance": [self.balance],
            "housing": [self.housing],
            "loan": [self.loan],
            "duration": [self.duration],
            "pdays": [self.pdays],
            "previous": [self.previous],
            "poutcome": [self.poutcome]
            
            }
            return pd.DataFrame(custom_data_input_dict) 
        
        except Exception as e:
            raise CustomException(e,sys)
   
    
       
    
    
# these funct return input in the form of dataframe 
# inshort from our  web application  whatever input is comming same input we are get mapp with the our abouve particaular val 