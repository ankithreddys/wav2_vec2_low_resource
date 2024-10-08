from feature_extraction.features import load_wav_feature

class Dataset:
    def __init__(self, dataframe, sr, processor, transform=None):
        self.dataframe = dataframe 
        self.sr = sr
        self.transform = transform
        self.processor = processor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        data = self.dataframe.iloc[idx]
        audio_feature = load_wav_feature(data, sr = self.sr)
        batch = dict()
        y = self.processor(audio_feature.reshape(-1), sampling_rate=16000).input_values[0] 
        batch["input_values"] = y
        with self.processor.as_target_processor():
            batch["labels"] = self.processor(data['transcription']).input_ids
        return batch