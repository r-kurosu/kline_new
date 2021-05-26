import numpy as np
import pandas as pd


def Read_other(FileName1, FileName2, FileName3, FileName4, FileName5, FileName6, Hold_encode):

    Ml_Load = pd.read_csv(FileName1)
    Ml_Back = pd.read_csv(FileName2)
    Ml_Afr = pd.read_csv(FileName3)
    Stress = pd.read_csv(FileName4)
    g_2 = pd.read_csv(FileName5)
    g_3 = pd.read_csv(FileName6)

    for i in range(len(Hold_encode)):
        g_2 = g_2.replace(Hold_encode.iloc[i, 0], Hold_encode.iloc[i, 1])
        g_3 = g_3.replace(Hold_encode.iloc[i, 0], Hold_encode.iloc[i, 1])
        Ml_Load = Ml_Load.replace(
            Hold_encode.iloc[i, 0], Hold_encode.iloc[i, 1])
        Ml_Back = Ml_Back.replace(
            Hold_encode.iloc[i, 0], Hold_encode.iloc[i, 1])
        Ml_Afr = Ml_Afr.replace(Hold_encode.iloc[i, 0], Hold_encode.iloc[i, 1])

    Stress = list(Stress.iloc[:, 1])

    GANG2 = []
    for n in range(2):
        add = []
        for k in range(len(g_2.iloc[:, n])):
            if str(g_2.iloc[k, n]) != 'nan':
                add.append(int(g_2.iloc[k, n]))
        GANG2.append(add)

    GANG3 = []
    for n in range(3):
        add = []
        for k in range(len(g_3.iloc[:, n])):
            if str(g_3.iloc[k, n]) != 'nan':
                add.append(int(g_3.iloc[k, n]))
        GANG3.append(add)

    return Ml_Load, Ml_Back, Ml_Afr, Stress, GANG2, GANG3
