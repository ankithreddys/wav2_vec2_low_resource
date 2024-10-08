from torch_dataloading.torch_dataset import Dataset
import json
from typing import Dict
import pandas as pd
import sys
sys.path.append('..')

class Dataset_creation(Dataset):
    def __init__(self,sr,path,special_tokens,processor):
        self.dataframe = self.load_dataset(path)
        self.sr = sr
        self.path = path
        self.special_tokens = special_tokens
        self.processor = processor

    def load_dataset(self,path):
        df = pd.read_csv(path)
        return df

    def dataset_loading(self) -> Dataset:
        dataset = Dataset(self.dataframe,self.sr,self.processor)
        return dataset

    def vocab_dict(self) -> Dict[int, str]:
        with open('/wav2vec2_assamese/DATASETS/vocab_assamese_new.json','r') as f:
            vocab_dict = json.load(f)
        '''vocab_dict["|"] = len(vocab_dict)
        #del vocab_dict[" "]
        for v in self.special_tokens.values():
            vocab_dict[v] = len(vocab_dict)
        with open('/wav2vec2_assamese/DATASETS/vocab_assamese_new.json', "w") as f:
            json.dump(vocab_dict, f)
        '''