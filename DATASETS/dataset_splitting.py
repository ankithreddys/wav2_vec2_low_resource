import pandas as pd
from sklearn.model_selection import train_test_split


def combine_datasets(conversation,read):
    CONVERSATION_CLEAR_WITHOUT_TIME = pd.read_csv(conversation)
    READ_CLEAR = pd.read_csv(read)
    new_df = pd.concat([CONVERSATION_CLEAR_WITHOUT_TIME,READ_CLEAR])
    new_df.reset_index(drop=True,inplace=True)
    return new_df

def split():
    dataset = combine_datasets('CONVERSATION_CLEAR_WITHOUT_TIME.csv','READ_CLEAR.csv')
    train,Test = train_test_split(dataset,test_size = 0.15)
    Train,Valid = train_test_split(train,test_size = 0.15)
    Train.to_csv('train.csv',index = False)
    Test.to_csv('test.csv',index = False)
    Valid.to_csv('validation.csv',index = False)
