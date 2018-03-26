import glob

import os
import pandas as pd


def unpack_zip(input_file_path, output_dir_path):
    import zipfile
    zip_ref = zipfile.ZipFile(input_file_path, 'r')
    zip_ref.extractall(output_dir_path)
    zip_ref.close()


def get_magic_number(file_path):
    with open(file_path, 'rb') as fd:
        file_head = fd.read(3)
        return file_head


def get_df_from_directory_files(path):
    from os import listdir
    from os.path import isfile, join
    data_files = [os.path.join(path, f) for f in listdir(path) if isfile(join(path, f))]
    list_ = []
    for file_ in data_files:
        df = pd.read_csv(file_, index_col=None, header=0, sep=",", names=["a", "b", "c"])
        list_.append(df)
    result_df = pd.concat(list_)
    return result_df
