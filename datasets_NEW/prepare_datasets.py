import pandas as pd


def prepare_dataframe():
    required_columns = ["path","sentence"]
    datasets = ["train.tsv","test.tsv","validated.tsv","other.tsv","invalidated.tsv","dev.tsv"]
    df = pd.DataFrame(columns=["path","sentence"])
    for x in datasets:
        df_1 = pd.read_csv(x,delimiter='\t')
        df_1 = df_1[required_columns]
        df = pd.concat([df_1,df],axis=0,ignore_index=True)
    df = df.drop_duplicates()
    df.to_csv("dataset.csv",index = False)


if __name__ == "__main__":
    prepare_dataframe()