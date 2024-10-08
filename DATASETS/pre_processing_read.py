import re
import os
import json
import pandas as pd
import torchaudio
import numpy as np

def remove_initial_spaces(text):
  return re.sub(r"(^\s+)(.*)", r"\2", text, flags=re.MULTILINE)

def dataset_load(dir_path):
    dataset = {}
    pattern = r"^\d+"
    root_dir = dir_path
    temp = 0
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.json'):
                with open((os.path.join(root,filename)),'r') as f:
                    data = json.load(f)
                
                    for key,_ in data.items():
                        if data[key]['text_transcription']:
                            dataset[temp] = {}
                            dataset[temp]['transcription'] = data[key]['text_transcription']
                            data[key]['audio_path'] = re.sub(pattern, "",data[key]['audio_path'] )
                            data[key]['audio_path'] = root + data[key]['audio_path'] + '.wav'
                            dataset[temp]['audio_path'] = data[key]['audio_path']
                            info = torchaudio.info(data[key]['audio_path'])
                            dataset[temp]['duration'] = np.round((info.num_frames / info.sample_rate),2)
                            temp += 1
    return dataset


def dict_store_as_json(output_file_name,dataset):
    with open(output_file_name, "w") as f:
        json.dump(dataset, f)

def json_store_as_csv(output_file_name,json_dataset_path):
    dataset = pd.read_json(json_dataset_path)
    dataset = dataset.transpose()
    dataset.dropna(inplace=True)
    dataset.reset_index(drop=True,inplace=True)
    dataset.to_csv(output_file_name,index = False)

def dataframe_store_as_json(output_file_name,dataframe):
    dataframe.to_json(output_file_name)

def dataframe_store_as_csv(output_file_name,json_dataset_path):
    dataset = pd.read_json(json_dataset_path)
    cols = ['transcription','audio_path','duration']
    dataset = dataset[cols]
    dataset = dataset[dataset['duration'] > 2]
    dataset['transcription'] = dataset['transcription'].apply(lambda x : remove_initial_spaces(x))
    dataset.dropna(inplace=True)
    indices_read = dataset[dataset['transcription'] == ' '].index
    dataset = dataset.drop(indices_read)
    indices_read = dataset[dataset['transcription'] == ''].index
    dataset = dataset.drop(indices_read)
    indices_read = dataset[dataset['transcription'].isna()].index
    dataset = dataset.drop(indices_read)
    dataset.reset_index(drop=True,inplace=True)
    dataset.to_csv(output_file_name,index = False)

def rm_invalid(x):
    #as unicode characters are not present for assamese alphabet is extracted from corpus and unnecessary characters are removed
    rm_chars = ['!','"','#','%','&',"'",'(',')','*',',','-','.','/','1',':',';','<','=','>','?','A','B','C','D','F',
                'I','L','M','N','O','P','S','T','U','V','`','a','b','c','d','e','f','g','h','i','k','l','m','n','o',
                'p','q','r','s','t','u','v','w','x','y','z','~','\x7f','ا','‘','’','\u200c','\u200d','_']
    as_number_unicodes = range(0x09e6,0x9ef+1)
    rm_chars.append(as_number_unicodes)
    return "".join([c for c in x['transcription'] if c not in rm_chars])
                

def dataset_creation(dir_path):
    dataset = dataset_load(dir_path)
    dict_store_as_json('READ_UNCLEAR.json',dataset)
    json_store_as_csv('READ_UNCLEAR.csv','READ_UNCLEAR.json')
    df = pd.read_csv('READ_UNCLEAR.csv')
    df['transcription'] = df.apply(rm_invalid,axis = 1)
    dataframe_store_as_json('READ_CLEAR.json',df)
    dataframe_store_as_csv('READ_CLEAR.csv','READ_CLEAR.json')
    return pd.read_csv('READ_CLEAR.csv')