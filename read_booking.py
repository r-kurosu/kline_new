import numpy as np
import pandas as pd


def Read_booking(FileName):

    Booking_df = pd.read_csv(FileName)

    L = []  # 積み地の集合
    D = []  # 揚げ地の集合
    T = []  # 港の集合

    lport_name = [x for x in list(
        Booking_df["PORT_L"].unique()) if not pd.isnull(x)]
    dport_name = [x for x in list(
        Booking_df["PORT_D"].unique()) if not pd.isnull(x)]

    for t in range(len(lport_name)):
        L.append(t)

    for t in range(len(dport_name)):
        D.append(t + max(L) + 1)

    T = L + D

    Check_port = []
    key1 = Booking_df.columns.get_loc('CPORT')
    for j in range(len(Booking_df)):
        count = 0
        if str(Booking_df.iloc[j, key1]) != 'nan':
            for p in lport_name:
                if p == str(Booking_df.iloc[j, key1]):
                    Check_port.append(count)
                else:
                    count = count + 1

    # 港番号のエンコード
    for k in range(len(lport_name)):
        Booking_df = Booking_df.replace(lport_name[k], L[k])

    for k in range(len(dport_name)):
        Booking_df = Booking_df.replace(dport_name[k], D[k])

    # 巨大な注文の分割
    divided_j = []
    divide_dic = []
    divide_df = Booking_df.iloc[0, :]
    SMALL_UNIT = 20
    for j in range(len(Booking_df)):
        unit = Booking_df.iloc[j, Booking_df.columns.get_loc('Units')]
        if 80 < unit:
            divied_u_num = int(unit // SMALL_UNIT)
            if (unit % SMALL_UNIT) != 0:
                divied_u_num += 1
            for i in range(divied_u_num):
                concat = Booking_df.iloc[j, :]
                concat["Units"] = SMALL_UNIT
                if (unit % SMALL_UNIT) != 0 and i==(divied_u_num-1):
                    concat["Units"] = unit % SMALL_UNIT
                divide_df = pd.concat([divide_df, concat], axis=1)
                divided_j.append(j)
                divide_dic.append([j, unit])
    divide_df = divide_df.T
    divide_df = divide_df.drop(divide_df.index[[0]])

    if len(divided_j) > 0:
        Booking_df = Booking_df.drop(Booking_df.index[divided_j])
        Booking = pd.concat([Booking_df, divide_df])
    else:
        Booking = Booking_df

    Booking["Index"] = 0
    for k in range(len(Booking)):
        Booking.iloc[k, Booking.columns.get_loc('Index')] = k

    A = list(Booking["RT"])
    J = list(Booking["Index"])
    U = list(Booking["Units"])
    G = list(Booking["Weight"])

    key2 = Booking.columns.get_loc('LPORT')
    key3 = Booking.columns.get_loc('DPORT')
    Port = Booking.iloc[:, key2:key3+1]

    # 注文をサイズ毎に分類
    J_small = []
    J_medium = []
    J_large = []
    for j in J:
        if U[j] <= 100:
            J_small.append(j)
        if 100 < U[j] and U[j] <= 200:
            J_medium.append(j)
        if 200 < U[j]:
            J_large.append(j)

    return T, L, D, J, U, A, G, J_small, J_medium, J_large, Port, Check_port, Booking, divide_dic
