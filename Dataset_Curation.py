import pandas as pd
from sklearn.model_selection import train_test_split
import re

scp_csv = pd.read_csv('D:\\Datasets\\test_scp_store.csv')
scp_csv = scp_csv.iloc[1:,:]

scp_full_train, scp_test = train_test_split(scp_csv, test_size = 0.1, random_state = 42069)
scp_train, scp_eval = train_test_split(scp_full_train, test_size = 0.1, random_state = 42069)

def build_dataset(df, dest_path):
    f = open(dest_path, 'w', encoding='utf-8')
    desc_data = ''
    desc = df['Description'].tolist()
    for scp in desc:
        scp = str(scp)[1:-2]
        scp = str(scp).strip()
        scp = re.sub(r"\s", " ", scp)
        bos_token = '<BOS>'
        eos_token = '<EOS>'
        desc_data += bos_token + ' ' + scp + ' ' + eos_token + '\n'
        
    f.write(desc_data)

path_desc_train = 'C:\\Users\\Azaghast\\GPT2_Finetune\\desc_train.txt'
path_desc_eval  = 'C:\\Users\\Azaghast\\GPT2_Finetune\\desc_eval.txt'
path_desc_test = 'C:\\Users\\Azaghast\\GPT2_Finetune\\desc_test.txt'

build_dataset(scp_train, path_desc_train)
build_dataset(scp_eval, path_desc_eval)
build_dataset(scp_test, path_desc_test)