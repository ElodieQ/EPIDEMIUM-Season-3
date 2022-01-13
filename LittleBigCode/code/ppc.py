"""
Preprocessing related functions
"""
import os 
import pandas as pd
from PIL import Image
from pathlib import Path
import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from functools import partial
import warnings
warnings.filterwarnings('ignore')



def _as_date(x):
    """ Helper to cast DataFrame date column """
    return datetime.datetime.strptime(x, "%Y-%m-%d")


def read_korl_csv(path):
    """ Read the KORL csv and potentially correct stuff """
    df = pd.read_csv(path)
    df['computed_os'] = df.apply(lambda row: \
            (_as_date(row['Date_derniere_nouvelles']) - _as_date(row['Date_biopsie'])).days / 30., axis=1)
    return df


def _get_id(x):
    """ Get patient ID from image file path """
    return str(x).split(os.sep)[-1].split('_')[0]

def get_id2f(markers_dpath):
    """ Find all images' paths for each patient """
    id2f = {}
    for i, dpath in enumerate(markers_dpath):
        fpaths = list(dpath.iterdir())
        for path in fpaths:
            _id = _get_id(path)
            if _id in id2f:
                id2f[_id].append(str(path))
            else:
                id2f[_id] = [str(path)]
    return id2f


def get_all_combinations(fpaths):
    """
    Produce all possible combinations of images for each patient, following
    the rule of 1 image per marker for each patient.
    """
    subsets = []
    for subset in itertools.combinations(fpaths, 6):
        skip = False
        markers = set(int(e.split('marker')[1].split(os.sep)[0]) for e in subset)
        for i in range(1, 7):
            if i not in markers:
                skip = True
                break
        if skip:
            continue
        subsets.append(tuple(sorted(subset)))
    return set(subsets)


def prepare_target(x):
    """ Encode the OS into 3 categories """
    if x <= 24:
        return 0
    elif x <= 72:
        return 1
    else:
        return 2


def prepare_dataset(db_path, id2f, is_train=True):
    """
    Read KORL csv files and produce the dataset : one sample contains 1 image of
    each marker for each patient. The dataset contains all combinations for each
    patient.

    Parameters
    --------
    db_path: str
        Path of the 'data/' directory
    id2f: dict
        Patient ID to list of images' paths dictionary
    is_train: bool
        Whether we expect a target column or not

    Returns
    --------
    df_full: pandas DataFrame
        Dataset
    """
    # Read csv
    df = read_korl_csv(db_path)
    ids = set(df['Patient_ID'].values.tolist())
    if is_train:
        id2os = {k: v for k, v in df[['Patient_ID', 'OS']].values.tolist()}
    else:
        df.iloc[0,0] = "905e61"  # Error in data
    # Get usable dataframe
    df_full = pd.DataFrame()
    for patient, fpaths in id2f.items():
        if patient not in ids:
            continue
        combinations = get_all_combinations(fpaths)
        cur_df = pd.DataFrame([[patient] + list(tup) for tup in combinations],
                     columns=['patient']+[f'marker{i}' for i in range(1,7)])
        df_full = pd.concat([df_full, cur_df], axis=0).reset_index(drop=True)
    if is_train:
        df_full['OS'] = df_full['patient'].apply(lambda x: id2os[x])
        df_full['target'] = df_full['OS'].apply(prepare_target)
    return df_full


def _split_train_val(df, test_size=.3):
    """
    Split the training dataframe into actual training and validation.
    Splitting based on patient ID

    Parameters
    --------
    test_size: float [0., 1.]
        Part of training patients (not samples !) to use as validation

    Returns
    --------
    df_train: pandas DataFrame
        Training data
    df_val: pandas DataFrame
        Validation data
    """
    id_train, id_val = train_test_split(df['patient'].unique(),
                                        test_size=.3,
                                        random_state=42)
    df_train = df[df['patient'].isin(id_train)].reset_index(drop=True)
    df_val = df[df['patient'].isin(id_val)].reset_index(drop=True)
    return df_train, df_val


def get_train_val_test_dfs(val_size=.3):
    """
    Gather the training and test data without loading images + create a
    validation set based on the training data.

    Parameters
    --------
    val_size: float [0., 1.]
        Part of training patients (not samples !) to use as validation

    Returns
    --------
    df_train: pandas DataFrame
        Training data
    df_val: pandas DataFrame
        Validation data
    df_test: pandas DataFrame
        Test data
    """
    # Constants
    data_path = Path('.').resolve().parents[0].joinpath('data')
    train_db_path = str(data_path.joinpath('KORL_avatar_train.csv'))
    test_db_path = str(data_path.joinpath('KORL_avatar_test_X.csv'))
    markers_dpath = [data_path.joinpath(f'marker{i}') for i in range(1, 7)]
    #
    id2f = get_id2f(markers_dpath)
    df_train = prepare_dataset(train_db_path, id2f, is_train=True)
    df_train, df_val = _split_train_val(df_train, test_size=val_size)
    df_test = prepare_dataset(test_db_path, id2f, is_train=False)
    return df_train, df_val, df_test

def red_count_preprocess(df, red_thresh=50):
    """
    Produce a dataframe of size N x 6, where N is the number samples and 6 is the
    6 different markers. Each value is the percentage of red pixels in each
    image.

    Parameters
    --------
    df: pandas DataFrame
        Dataset with unloaded images, contains the images' paths for each sample
    red_thresh: int [0,255]
        Value above which the pixel is considered red

    Returns
    --------
    df : pandas Dataframe
        Datframe with 6 columns ( 'marker_1', ..., 'marker_6)
    """
    img2red = {}
    # Function for each row
    def _df_to_img(row):
        img = []
        for i in range(1, 7):
            fpath = row[f"marker{i}"]
            if fpath in img2red:
                img.append(img2red[fpath])
            else:
                tmp = cv2.imread(row[f"marker{i}"])[:,:,0]
                tmp[tmp[:,:]<red_thresh] = 0
                tmp[tmp[:,:]>0] = 1
                res = np.sum(tmp) / (1404*1872)
                img.append(res)
                img2red[fpath] = res
        return img
    X = np.array(df.apply(_df_to_img, axis=1).values.tolist())
    df = pd.DataFrame( X, columns = ['marker_{}'.format(i) for i in range(1, 7)], index = df['patient'])
    return df



def preprocess_KORL (features, db_path ) : 
    """
    Produce a dataframe of size N_patient x features, where N is of patient in the clinical data. 
    image.

    Parameters
    --------
    features: list
         List of columns of the clinical data to keep
    db_path : string
        Path of the clinical data

    Returns
    --------
    df : panda dataframe
        Datframe with len(features) columns
    """
    #Read and preprocess data
    df = pd.read_csv(db_path)
    df = df.set_index("Patient_ID")
    df['N'] = df['N'].replace(to_replace=r'^2[a,b,c]', value='2', regex=True).astype(int)
    df['Age_diag'] = round(df['Age_diag']/10).astype(int)
    return  df[features]

def full_preprocess(features, db_path, df, red_thresh= 50 ) :
    """
    Produce a dataframe of size N x 6 + len(features), where N is the number samples, 6 is the
    6 different markers. Each value for the market is the percentage of red and there is also the clinical data.
    image.

    Parameters
    --------
    features: list
        List of columns of the clinical data to keep
    df_path : string
        Path of the clinical csv data
    df: pandas DataFrame
        Dataset with unloaded images, contains the images' paths for each sample
    red_thresh: int [0,255]
        Value above which the pixel is considered red

    Returns
    --------
    df_final : pandas dataframe
        Datframe with the 6 columns ( 'marker_1', ..., 'marker_6) and the features columns from the clinical data
    """
    df_images = red_count_preprocess(df, red_thresh)
    df_clinical = preprocess_KORL(features, db_path)
    df_final = pd.merge(df_images, df_clinical, left_index= True, right_index= True, how = 'inner')
    return df_final