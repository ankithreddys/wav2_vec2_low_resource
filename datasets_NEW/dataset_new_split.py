import pandas as pd
from sklearn.model_selection import train_test_split

def dataset_split():
    dataset = pd.read_csv("dataset.csv")
    train,test = train_test_split(dataset,test_size = 0.15)
    train, validation = train_test_split(train,test_size = 0.15)
    train.to_csv('train.csv',index=False)
    test.to_csv('test.csv',index = False)
    validation.to_csv('validation.csv',index=False)


if __name__ == "__main__":
    dataset_split()