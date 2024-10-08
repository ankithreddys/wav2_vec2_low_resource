import os
import json
import re
import pandas as pd
import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
import datetime

def remove_initial_spaces(text):
  return re.sub(r"(^\s+)(.*)", r"\2", text, flags=re.MULTILINE)


def dataset_load(dir_path):
    dataset = {}
    temp = 0
    pattern_transcription = r"TRANSCRIPTION$"
    pattern_removejson = r".json$"
    root_dir = dir_path
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.json') and "merged" in filename:
                with open((os.path.join(root,filename)),'r') as f:
                    data = json.load(f)

                    for i in range(len(data['value']['segments'])):
                        if "transcriptionData" in data['value']['segments'][i]:
                            dataset[temp] = {}
                            dataset[temp]['transcription'] = (data['value']['segments'][i]['transcriptionData']['content'])
                            dataset[temp]['start_time'] = data['value']['segments'][i]['start']
                            dataset[temp]['end_time'] = data['value']['segments'][i]['end']
                            dataset[temp]['audio_path'] = re.sub(pattern_transcription,'',root) + 'AUDIO/' + re.sub(pattern_removejson,'',filename) + '.wav'
                            dataset[temp]['duration'] = float(dataset[temp]['end_time'])-float(dataset[temp]['start_time'])
                            temp += 1
    return dataset



def dict_store_as_json(output_file_name,dataset):
    with open(output_file_name, "w") as f:
        json.dump(dataset, f)
        f.close()

def json_store_as_csv(output_file_name,json_dataset_path):
    dataset = pd.read_json(json_dataset_path)
    dataset = dataset.transpose()
    wrong_index = dataset[dataset['duration']<0].index
    dataset.drop(wrong_index,axis=0,inplace=True)
    dataset.to_csv(output_file_name,index = False)

def dataframe_store_as_json(output_file_name,dataframe):
    dataframe.to_json(output_file_name)

def dataframe_store_as_csv_withtime(output_file_name,json_dataset_path):
    dataset = pd.read_json(json_dataset_path)
    cols = ['transcription','start_time','end_time','audio_path','duration']
    dataset = dataset[cols]
    dataset = dataset[dataset['duration'] > 2]
    dataset['transcription'] = dataset['transcription'].apply(lambda x : remove_initial_spaces(x))
    dataset.to_csv(output_file_name,index = False)

def dataframe_store_as_csv_withouttime(output_file_name,dataset_path):
    dataset = pd.read_csv(dataset_path)
    cols = ['transcription','audio_path','duration']
    dataset = dataset[cols]
    dataset.dropna(inplace=True)
    indices_conversation = dataset[dataset['transcription'] == ' '].index
    dataset = dataset.drop(indices_conversation)
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
    dict_store_as_json('CONVERSATION_UNCLEAR.json',dataset)
    json_store_as_csv('CONVERSATION_UNCLEAR.csv','CONVERSATION_UNCLEAR.json')
    df = pd.read_csv('CONVERSATION_UNCLEAR.csv')
    df['transcription'] = df.apply(rm_invalid,axis = 1)
    dataframe_store_as_json('CONVERSATION_CLEAR.json',df)
    dataframe_store_as_csv_withtime('CONVERSATION_CLEAR_WITH_TIME.csv','CONVERSATION_CLEAR.json')
    dataframe_store_as_csv_withouttime('CONVERSATION_CLEAR_WITHOUT_TIME.csv','CONVERSATION_CLEAR_WITH_TIME.csv')
    return pd.read_csv('CONVERSATION_CLEAR_WITHOUT_TIME.csv'), pd.read_csv('CONVERSATION_CLEAR_WITH_TIME.csv')
