import pandas as pd
#from pathlib import Path

##need to create a function that will extract text from file and return it as an xml docuemnt ready for analysis

##def extract_text_to_xml(file_path):
##    #extract text and converting to XML
##    pass

input_csv = r"G:\My Drive\Birkbeck\Project\Hansard\hansard-speeches-v310.csv"

sample = pd.read_csv(input_csv, nrows=5)

#print(sample)
print(list(sample.columns))
