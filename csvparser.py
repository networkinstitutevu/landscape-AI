import pandas as pd


def get_csv(file_path):
    """
    Inputs a CSV file and return a pandas dataframe.
    :param file_path:
    :return:
    """


    file_df = pd.read_csv(file_path)
    if 'generative_type' not in file_df.columns.values:
        file_df['generative_type'] = ''

    if 'is_generative' not in file_df.columns.values:
        file_df['is_generative'] = ''

    # cast all columns in the dataframe to string
    return file_df.astype(str)
