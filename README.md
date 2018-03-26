# Test task: clusterization

Repository for clusterization task
* **input** - Demo.dim
* **task** - make non-hierarchical clusterization of 3D dataset
* **output** - python 3 compatible code

## step 0: You can see some results in ```report.ipynb```

## step 1: Data extraction

### Figuring out of data type

Every file has its magic number. 
So with function **get_magic_number** I get that magic number of 'Demo.bin' is **PK\x03**.
It's zip's file magic number, so I unzipped Demo.bin, and received text data.
Get pandas dataframe with method **get_df_from_directory_files**.
Unzipping, df making and magic number functions are in **data_extraction.py** file.


## step 2: Data preprocessing

* ### Remove outliers
* Use ```pd.DataFrame.hist()``` for getting information about distribution
* remove outliers - method ```data_preprocessing.remove_outliers```

* ### Normalization
* Use sklearn methods for minmax normalization

## step3: Clusterization

I tried few approaches for K-means:</br>
* with one-hot encoding - because of third column in data
* without one-hot encoding

For getting number of clusters I used silhouette analysis.

**My optimal number of clusters is 6.**</br>
Because in this case we can see classes on the different levels of the z-axis.


