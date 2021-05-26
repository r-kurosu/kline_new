import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def Read_hold(FileName):

    Hold_df = pd.read_csv(FileName)

    B = list(Hold_df["Resourse"])
    RT_benefit = list(Hold_df["RT_benefit"])
    delta_s = list(Hold_df["Weight_s"])
    delta_h = list(Hold_df["Weight_h1"] * Hold_df["Weight_h2"])
    min_s = Hold_df.iloc[0, Hold_df.columns.get_loc('Min_s')]
    max_s = Hold_df.iloc[0, Hold_df.columns.get_loc('Max_s')]
    max_h = Hold_df.iloc[0, Hold_df.columns.get_loc('Max_h')]

    list_drop_cols = ['Resourse', 'RT_benefit', 'Weight_s',
                      'Weight_h1', 'Weight_h2', 'Min_s', 'Max_s', 'Max_h']

    # ホールド番号のエンコード
    Hold_encode = Hold_df.iloc[:, 0:2]
    le = LabelEncoder()
    Hold_encode["Resourse"] = le.fit_transform(Hold_df['Hold'].values)
    Hold_encode = Hold_encode.rename(columns={'Resourse': 'Index'})

    Hold_data = Hold_df.drop(list_drop_cols, axis=1)

    for i in range(len(Hold_encode)):
        Hold_data = Hold_data.replace(
            Hold_encode.iloc[i, 0], Hold_encode.iloc[i, 1])

    I = list(Hold_data["Hold"])
    I_pair = []
    I_next = []
    I_same = []
    I_lamp = []
    I_deck = []

    key1 = Hold_data.columns.get_loc('Pair_Hold1')
    key2 = Hold_data.columns.get_loc('Next_Hold1')
    key3 = Hold_data.columns.get_loc('Same_Hold1')
    key4 = Hold_data.columns.get_loc('Lamp_Hold')
    key5 = Hold_data.columns.get_loc('Region1')

    for i in range(len(Hold_data)):
        if str(Hold_data.iloc[i, key1]) != 'nan':
            I_pair.append([int(Hold_data.iloc[i, key1]),
                           int(Hold_data.iloc[i, key1+1])])
        if str(Hold_data.iloc[i, key2]) != 'nan':
            I_next.append([int(Hold_data.iloc[i, key2]),
                           int(Hold_data.iloc[i, key2+1])])
        if str(Hold_data.iloc[i, key3]) != 'nan':
            I_same.append([int(Hold_data.iloc[i, key3]),
                           int(Hold_data.iloc[i, key3+1])])
        if str(Hold_data.iloc[i, key4]) != 'nan':
            I_lamp.append(int(Hold_data.iloc[i, key4]))

    last = len(Hold_data.T)
    for i in range(key5, last):
        append_list = []
        count = 0
        while str(Hold_data.iloc[count, i]) != 'nan':
            append_list.append(int(Hold_data.iloc[count, i]))
            count = count + 1
        I_deck.append(append_list)

    return I, B, I_pair, I_next, I_same, I_lamp, I_deck, RT_benefit, delta_s, min_s, max_s, delta_h, max_h, Hold_encode, Hold_df
