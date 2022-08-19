from turtle import pd
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class DataEngineer :
    
    def __init__(self, data: pd.DataFrame) -> None:
        self.df = data
        
    def remove_unnamed(self) -> None:
        self.df = self.df.drop('Unnamed: 2', axis=1)
        self.df = self.df.drop('Unnamed: 3', axis=1)
        self.df = self.df.drop('Unnamed: 4', axis=1)
        
    def rename_columns(self) -> None:
        self.df = self.df.rename(
            columns = {
                'v1' : 'label',
                'v2' : 'text'
            }
        )
        
    def encode_labels(self) -> None:
        le = LabelEncoder()
        
        self.df['numeric_labels'] = le.fit_transform(self.df.label.values)
        
        
    def return_cleaned_data(self) -> pd.DataFrame:
        return self.df     