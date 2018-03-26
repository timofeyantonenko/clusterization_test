import re

import numpy as np
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, silhouette_samples


def process_c_column_values(x):
    try:
        return int(x)
    except ValueError:
        splitted = re.split("-+", x)
        splitted = list(map(int, splitted))
        return sum(splitted) / len(splitted)


def preprocessing_of_data(input_df):
    # print(input_df.c.value_counts())
    input_df['c'] = input_df['c'].apply(lambda x: process_c_column_values(x))
    input_df = input_df[input_df["c"] != 1]
    # scaler = preprocessing.scale(input_df)
    # scaler = StandardScaler().fit(input_df)
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(input_df)

    return scaler


def add_one_hot_encoding(input_df, categorical_labels_list):
    le = preprocessing.LabelEncoder()

    only_categorical_columns_df = input_df[categorical_labels_list]
    transformed_to_labels_df = only_categorical_columns_df.apply(le.fit_transform)

    enc = preprocessing.OneHotEncoder()
    enc.fit(transformed_to_labels_df)
    one_hot_labels = pd.DataFrame(enc.transform(transformed_to_labels_df).toarray())

    non_categorical_columns = list(set(input_df.columns) - set(categorical_labels_list))

    result = pd.concat([input_df[non_categorical_columns], one_hot_labels], axis=1, join='inner')

    return result, enc


def turn_back_one_hot_encoding(one_hot_columns, one_hot_encoder):
    decoded = one_hot_columns.dot(one_hot_encoder.active_features_).astype(int)
    return decoded


def make_silhouette_analysis(input_data, range_n_clusters):
    plt.rcParams["figure.figsize"] = [12, 9]
    for n_clusters in range_n_clusters:

        fig = plt.figure(figsize=plt.figaspect(2.), dpi=300)
        fig.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        # First subplot
        ax1 = fig.add_subplot(2, 2, 1)
        # Second subplot
        ax2 = fig.add_subplot(2, 1, 2, projection='3d')

        # ax2 = Axes3D(fig3D)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(input_data) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(input_data)

        silhouette_avg = silhouette_score(input_data, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(input_data, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(input_data[:, 0], input_data[:, 1], input_data[:, 2], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], c[2], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        ax2.set_zlabel("Feature space for the 3rd feature")

        plt.show()


def remove_outliers(input_data):
    # input_data.hist()
    # plt.show()
    # see, that b has normal distribution
    # now can remove outliers by 3-sigma rule

    mean = input_data.b.mean()
    std = input_data.b.std()

    input_data = input_data[input_data.b <= mean + 2*std]
    input_data = input_data[input_data.b >= mean - 2*std]
    return input_data


if __name__ == '__main__':
    pass
