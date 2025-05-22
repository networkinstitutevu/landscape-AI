import csv
import pandas as pd


def get_csv(file_path):
    """
    Inputs a CSV file and return a pandas dataframe.
    :param file_path:
    :return:
    """


    file_df = pd.read_csv(file_path)

    # cast all columns in the dataframe to string
    return file_df.astype(str)