import pandas as pd
import numpy as np

class DataExplorer:
    
    def __init__(self, data: pd.DataFrame) -> None:
        self.df = data
        
    def word_lengths(self) -> list:
        
        lengths = []
        
        for i in range(0, len(self.df)):
            lengths.append(len(self.df.loc[i]['text'].split()))
            
        return lengths
        
    def total_words(self) -> int:
        length = np.sum(self.word_lengths())
        
        return length
    
    def mean_sentence_length(self) -> int:
        mean_len = int(np.mean(self.word_lengths()))
        
        return mean_len