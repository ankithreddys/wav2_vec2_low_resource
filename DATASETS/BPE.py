import pandas as pd
from collections import defaultdict
import json
import csv



def corpus_prepare(#Dataset_READ_Path,Dataset_CONVERSATIONAL_Path
        ):
    '''df_read = pd.read_csv(Dataset_READ_Path)
    df_conversation = pd.read_csv(Dataset_CONVERSATIONAL_Path)
    corpus_read = ' '.join(df_read['transcription'])
    corpus_conversation = ' '.join(df_conversation['transcription'])
    corpus = corpus_read + ' ' + corpus_conversation
    corpus = corpus.split(' ')
    '''
    df = pd.read_csv('/wav2vec2_assamese/datasets_NEW/dataset.csv')
    corpus = ' '.join(df['transcription'])
    corpus = corpus.split(' ')
    return corpus

def word_freqs_splits(corpus):
    word_freqs = defaultdict(int)
    for text in corpus:
        word_freqs[text] += 1
    
    splits = {word: [c for c in word] for word in word_freqs.keys()}
    return splits,word_freqs


def compute_pair_freqs(splits,word_freqs):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def merge_pair(a, b, splits,word_freqs):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits



def core_BPE(vocab_size,splits,word_freqs):
    vocab = []

    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(splits,word_freqs)
        best_pair = ""
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        splits = merge_pair(*best_pair, splits,word_freqs)
        vocab.append(best_pair[0] + best_pair[1])

    return vocab



def vocab_extraction(vocab_number):
    corpus = corpus_prepare(#'READ_CLEAR.csv','CONVERSATION_CLEAR_WITHOUT_TIME.csv'
        )
    word_splits,word_freqs = word_freqs_splits(corpus)
    vocab = core_BPE(vocab_number,word_splits,word_freqs)
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab))}
    with open('vocab_assamese_new.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
        vocab_file.close()
    print('Vocab_extracted of length {0}'.format({len(vocab)}))
    return vocab_dict