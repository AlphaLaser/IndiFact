from concurrent.futures import process
from data_engineer import DataEngineer
import pandas as pd

data = pd.read_csv("Data/Raw/raw_data.csv")

processor = DataEngineer(data)

processor.remove_unnamed()
processor.rename_columns()
processor.encode_labels()

processed_data = processor.return_cleaned_data()

processed_data.to_csv("Data/Processed/processed_data.csv", index=False)