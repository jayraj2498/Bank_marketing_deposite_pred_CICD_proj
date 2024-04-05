import os 
import sys 

from src.exception import   CustomException 
from src.logger import logging 

from  src.utils import read_sql_data 

from sklearn.model_selection import train_test_split  

from dataclasses import dataclass 




@dataclass 
class DataIngestionConfig: 
    train_data_path: str= os.path.join('artifacts','train.csv')
    test_data_path: str= os.path.join('artifacts','test.csv')
    raw_data_path: str= os.path.join('artifacts','data.csv')   
    
    
class DataIngestion :
    def __init__(self) :
        self.ingestion_config=DataIngestionConfig()  
        
    
    def initiate_data_ingestion(self) :
        logging.info("entered the Data_ingestion method ")
        
        try:
            df=read_sql_data() 
            print(df.head(3) ) 
            logging.info("reading the dataset from mysql is completed in data_ingestion") 
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path) , exist_ok=True) 
            df.to_csv(self.ingestion_config.raw_data_path , index=False , header=True)  
            
            logging.info("Train test split Method started ") 
            
            train_set,test_set = train_test_split(df , test_size=0.21 , random_state=41) 
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) 
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Ingestion of the data iss completed ")
            
            return ( 
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                    )
              
        
        except Exception as e:
            raise Exception(e,sys) 