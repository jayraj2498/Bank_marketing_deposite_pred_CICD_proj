# here we give data to model to  

import os 
import sys 
from src.exception import CustomException 
from src.logger import  logging 

from dataclasses import dataclass 
import pandas as pd 
import numpy as np 

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

from src.utils import save_object  , evaluate_models



# here we save the model now 

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl") 
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig() 
        
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            
            logging.info("splitting train and testing data into trian_test_split:") 
            
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Logistic Regression": LogisticRegression(),
                "SVM": SVC(),
                # "GaussianNB":GaussianNB(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(),
                "CatBoost": CatBoostClassifier(verbose=False),
                "AdaBoost": AdaBoostClassifier()
            }
            
            
            # Hyperparameters for tuning
            params = {
                "Logistic Regression": {},
                "SVM": {
                    'kernel': ['linear', 'poly'],
                    'C': [0.1, 1],
                    'gamma': ['scale']
                },
                "Random Forest": {
                    'n_estimators': [100],
                    'max_depth': [10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [ 2 ]
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': [10]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [100],
                    'max_depth': [3]
                },
                "XGBoost": {
                    'learning_rate': [0.1],
                    'n_estimators': [100],
                    'max_depth': [3]
                },
                "CatBoost": {
                    'depth': [4],
                    'learning_rate': [0.01],
                    'iterations': [50]
                },
                "AdaBoost": {
                    'n_estimators': [50],
                    'learning_rate': [0.01]
                }
            } 
            
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, 
                                                models=models ,params=params) 
            
            
            ## To get best model score from dict
            best_model_score=max(sorted(model_report.values())) 
            
            ## To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            print("This is the best model:")
            print(best_model_name)
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model) 
            
            
            
            predicted=best_model.predict(X_test)
            
            A_S=accuracy_score(y_test, predicted)  
            
            return A_S
         
        except Exception as e:
            raise CustomException(e,sys) 