import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_extraction import get_magic_number, unpack_zip, get_df_from_directory_files
from data_preprocessig import preprocessing_of_data, add_one_hot_encoding, turn_back_one_hot_encoding, \
    make_silhouette_analysis, remove_outliers
from scikit_learn_clusterizations import sl_k_means

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


def get_data(path_to_file):
    if os.path.isfile(path_to_file):
        return pd.read_csv(path_to_file)
    else:
        df = get_df_from_directory_files(path_to_file.split(".")[0])
        df.to_csv(path_to_file, index=False)
        return df


def plot_data(x, classes_rows):

    fig = pyplot.figure()
    plt = Axes3D(fig)
    plt.set_xlabel('a')
    plt.set_ylabel('b')
    plt.set_zlabel('c')

    input_data = pd.DataFrame(x, columns=["a", "b", "c"])
    input_data["classes"] = classes_rows
    input_data = input_data.sample(3000)
    # y_pred = classes_rows[:1000]

    plt.scatter(input_data["a"], input_data["b"], input_data["c"], c=input_data["classes"])
    # plt.title("Clusterization")
    pyplot.show()


def main():
    data_file = "Demo.csv"
    path_to_data = "Demo.bin"

    path_to_unzipped = path_to_data.split(".")[0]
    # print(get_magic_number(path_to_data))
    # unpack_zip(path_to_data, path_to_unzipped)

    df = get_data(data_file)
    remove_outliers(df)

    # sl without encoding
    scaler = preprocessing_of_data(df)
    # sampled_df = df.sample(3000)
    # sampled_df = scaler.fit_transform(sampled_df)
    # make_silhouette_analysis(sampled_df, [2, 3, 4, 5, 6, 7, 8, 9])

    df = scaler.fit_transform(df)
    classes_data = sl_k_means(6, df)

    # sl with one hot encoding
    # result, one_hot_encoder = add_one_hot_encoding(df, ["c"])
    # scaler = MinMaxScaler(feature_range=(-1, 1)).fit(result)
    # learning_dataset = scaler.fit_transform(result)
    # classes_data = sl_k_means(12, learning_dataset)
    # one_hot_encoded_names = list(set(result.columns) - set(["a", "b"]))
    # one_hot_encoded_columns = turn_back_one_hot_encoding(result[one_hot_encoded_names], one_hot_encoder)
    # df = pd.concat([result[["a", "b"]], one_hot_encoded_columns], axis=1, join='inner')
    # df = df.rename(index=str, columns={0: "c"})
    # scaler = MinMaxScaler(feature_range=(-1, 1)).fit(df)
    # df = scaler.fit_transform(df)

    # plotting results
    plot_data(df, classes_data)


if __name__ == '__main__':
    main()
