import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import sqlite3
from Sql import SqlManager
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

def pre_processing(df: DataFrame):
    """
    input : a data frame
    outputs: clean data frame
            dtype.txt : a file that has type of each columns
            database:information.sqlite
            tables:
                 information  : clean data frame
                 before_process : data before process
                 missing_information : information of missing_data function output
                 outliers : outliers data
                 describe : describe of clean data

    Description:
                delete null information
                merge capital_gain and capital_loss
                delete education column
                delete outlier information with IQR method
                save information in database

    """
    col1 = ['status_type', 'num_comments', 'num_reactions',
            'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas',
            'num_sads', 'num_angrys']


    sql_manager = SqlManager("information.sqlite")
    df.to_sql(name="before_process", con=sql_manager.conn, if_exists="replace")

    # drop duplicates if any.
    df.drop_duplicates(inplace=True)

    # check for missing value
    missing_data_df = missing_data(df)
    missing_data_df.to_sql(name="missing_information", con=sql_manager.conn, if_exists="replace")

    # drop nonless columns
    main_df = df.drop(
        columns=['status_id', 'status_published', 'Column1', 'Column2', 'Column3', 'Column4'])

    # convert status_type to int value
    le = LabelEncoder()
    main_df['status_type'] = le.fit_transform(main_df['status_type'])

    # scale data
    main_df = StandardScaler().fit_transform(main_df)
    main_df = pd.DataFrame(main_df, columns=col1)

    print(main_df['status_type'])

    # store clean data in sql
    main_df.to_sql(name="information", con=SqlManager("information.sqlite").conn, if_exists="replace", index=False)

    # store data set information in sql
    main_df.describe().to_sql(name="describe", con=sql_manager.conn, if_exists='replace')

    with open("outs\\dtypes.txt", "w") as file:
        file.write(str(main_df.dtypes))

    return main_df


def missing_data(data):
    """
        information of missing data
    """
    total = data.isnull().sum()
    percent = (data.isnull().sum() / data.isnull().count() * 100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return np.transpose(tt)


if __name__ == '__main__':
    csv_file = "FaceBook-dataset.csv"
    df = pd.read_csv(csv_file, skipinitialspace=True)
    df = pre_processing(df=df)
