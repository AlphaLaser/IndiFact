import pandas as pd
from data_explorer import DataExplorer

data = pd.read_csv('Data\Processed\processed_data.csv')

exploration = DataExplorer(data)

word_lengths = exploration.word_lengths()
total_words = exploration.total_words()
msl = exploration.mean_sentence_length()

print(f"Word Lengths : {(str(word_lengths[0:5])[0:20]).replace(']', ' ...]')}")
print(f"Total Word Count : {total_words}")
print(f"Average no. of words per sentence : {msl}")