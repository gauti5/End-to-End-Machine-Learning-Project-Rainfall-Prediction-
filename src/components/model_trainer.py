import os 
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logging import logging

from src.utils import save_object, evaluate_model

from pathlib import Path
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('Artifacts', 'Model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting the data into training and testing data')
            
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
            'Logistic Regression' : LogisticRegression(),
            'Bernoulli NB' : BernoulliNB(),
            'Decision Tree Classifier' : DecisionTreeClassifier(),
            'SVM' : SVC(),
            'Random Forest Classifier' : RandomForestClassifier(),
            'Gradient Boosting Classifier' : GradientBoostingClassifier(),
            'AdaBoost Classifier': AdaBoostClassifier(),
            'K Neigbors Classifier': KNeighborsClassifier()
            }
            
            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            print(model_report)
            
            print("\n-------------------------------------------------------\n")
            
            logging.info(f"model report : {model_report}")
            
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=models[best_model_name]
            
            print(f"Best Model Found, model Name : {best_model_score}, f1_score:{best_model}")
            
            print("\n===============================================\n")
            
            logging.info(f"Best Model Found, model Name : {best_model_score}, f1_score:{best_model}")
            
            if best_model_score<0.7:
                raise CustomException("Best Model Not Found!!")
                
            logging.info("Best Model Found!!")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
        except Exception as e:
            logging.info("Exception occured during the model training!!")
            raise CustomException(e,sys)
                
        