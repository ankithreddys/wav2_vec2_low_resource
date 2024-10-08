import pandas as pd
import torchaudio
import torch

def is_multichannel(wav):
    if wav.shape[0] > 1:
        return True
    else:
        return False



def multi_to_single(wav):
    if is_multichannel(wav):
        wav = torch.mean(wav,dim = 1)
        return wav
    else:
        return wav



def load_wav_feature(x, sr):
    ###path of conversational with time dataset should be given
    #df = pd.read_csv('/wav2vec2_assamese/DATASETS/CONVERSATION_CLEAR_WITH_TIME.csv') 
    '''if x['audio_path'] in df['audio_path'].values:
        record = df[(df['audio_path'] == x['audio_path']) & (df['transcription'] == x['transcription'])]
        i = record.index[0]
        start,end = record['start_time'][i],record['end_time'][i]
        wav,osr = torchaudio.load(record['audio_path'][i])
        wav = multi_to_single(wav)
        start_index = int(start * osr)
        end_index = int(end * osr)
        wav = wav[0,start_index:end_index]
        wav = torchaudio.functional.resample(wav,osr,sr)
        return wav
    
    else:
    '''    
    wav, osr = torchaudio.load(x['audio_path'])
    wav = multi_to_single(wav)
    wav = torchaudio.functional.resample(wav,osr,sr)
    return wav