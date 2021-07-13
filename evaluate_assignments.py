import warnings
import openpyxl
import numpy as np
import pandas as pd
import gurobipy as gp
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
import sys
args = sys.argv
import read_booking
import read_hold
import read_other

warnings.filterwarnings("ignore")

BookingFile = "book/exp_height.csv"
AssignmentsFile = "result/exp_height_assignment.xlsx"
HoldFile = "data/hold.csv"
MainLampFile = "data/mainlamp.csv"
BackMainLampFile = "data/back_mainlamp.csv"
AfrMainLampFile = "data/afr_mainlamp.csv"
StressFile = "data/stress_mainlamp.csv"
Gang2File = "data/gangnum_2.csv"
Gang3File = "data/gangnum_3.csv"



T, L, D, J, U, A, G, J_small, J_medium, J_large, Port, Check_port, Booking, divide_dic\
    = read_booking.Read_booking(BookingFile)

# 船体情報の読み込み1
I, B, I_pair, I_next, I_same, I_lamp, I_deck, RT_benefit, delta_s, min_s, max_s, delta_h, max_h, Hold_encode, Hold\
    = read_hold.Read_hold(HoldFile)

 # 船体情報の読み込み2
Ml_Load, Ml_Back, Ml_Afr, Stress, GANG2, GANG3\
        = read_other.Read_other(MainLampFile, BackMainLampFile, AfrMainLampFile, StressFile, Gang2File, Gang3File, Hold_encode)


df = pd.read_excel(AssignmentsFile)
print(df)



# ハイパーパラメータ設定
# 各目的関数の重み
w1 = 1
w2 = 1
w3 = 1
w4 = 1
w5 = 1

# 目的関数1のペナルティ
penal1_z = 10

# 目的関数2のペナルティ
penal2_load = 1
penal2_dis = 10


# 目的関数5のペナルティ
penal5_k = 1000

order_num = len(set(df["Order_ID"]))

V_ij = []
for i in I:
    tmp = []
    for j in J:
        tmp.append(0)
    V_ij.append(tmp)
# print(len(V_ij))

OBJ = 0

for index, row in df.iterrows():
    # print(row)
    hold_idx = Hold_encode[Hold_encode["Hold"]==row["Hold_ID"]]["Index"].iloc[-1]
    
    # print(Booking[Booking["Order_num"] == int(row["Order_ID"])]["Index"].iloc[-1])
    order_idx = int(row["Order_ID"]-1)
    V_ij[hold_idx][order_idx] = row["Units"]

for i in range(len(V_ij)):
    print(V_ij[i])

# 目的関数1
# Z_it1t2:ホールドiにおいてt2を通過する注文があるとき異なる乗せ港t1の数分のペナルティ
OBJ1 = 0
for hold_idx in range(len(V_ij)):
    orders = V_ij[hold_idx]
    dport = []
    for i in range(len(orders)):
        if orders[i]>0:
            dport.append(df[df["Order_ID"]==i+1]["DPORT"].iloc[-1])
    dport = set(dport)
    if len(dport)>2:
        OBJ1 += w1 * penal1_z* (len(dport)-1)
print(OBJ1)

OBJ += w1 * OBJ1

# 目的関数2
OBJ2 = 0
for p in I_pair:
    hold1 = p[0]
    hold2 = p[1]
    orders1 = V_ij[hold1]
    orders2 = V_ij[hold2]
    lport1 = []
    lport2 = []
    dport1 = []
    dport2 = []
    for i in range(len(orders1)):
        if orders1[i]>0:
            dport1.append(df[df["Order_ID"]==i+1]["DPORT"].iloc[-1])
            lport1.append(df[df["Order_ID"]==i+1]["LPORT"].iloc[-1])
        if orders2[i]>0:
            dport2.append(df[df["Order_ID"]==i+1]["DPORT"].iloc[-1])
            lport2.append(df[df["Order_ID"]==i+1]["LPORT"].iloc[-1])
    dport1 = set(dport1)
    dport2 = set(dport2)
    dport = set()
    if len(dport.union(dport1,dport2)) > 1:
        OBJ2 += penal2_dis * len(dport.union(dport1,dport2))-1
        
    lport1 = set(lport1)
    lport2 = set(lport2)
    lport = set()
    if len(lport.union(lport1,lport2)) > 1:
        OBJ2 += penal2_load * len(lport.union(lport1,lport2))-1
        
print(OBJ2)

OBJ += w2 * OBJ2

# 目的関数3
OBJ3 = 0
check_port = L[:-1] + D[:-1]

for hold_idx in range(len(V_ij)):
    orders = V_ij[hold_idx]
    accpeted_rate = Stress[hold_idx]
    # print(accpeted_rate)
    order_array = []
    for loading_num in range(len(T)):    
        tmp = []
        for i in range(len(orders)):
            tmp.append(0)
        order_array.append(tmp)
    for i in range(len(orders)):
        if orders[i]>0:
            lport = df[df["Order_ID"]==i+1]["LPORT"].iloc[-1]
            for j in range(lport,L[-1]+1):
                order_array[j][i] = orders[i]
    
    for i in range(len(orders)):
        if orders[i]>0:
            dport = df[df["Order_ID"]==i+1]["DPORT"].iloc[-1]
            for j in range(D[0],dport):
                order_array[j][i] = orders[i]
            
            
    for port in check_port:
        total_RT = 0
        order = order_array[port]
        for i in range(len(order)):
            if order[i]>0:
                single_rt = df[df["Order_ID"]==i+1]["RT"].iloc[-1]
                total_RT += order[i]*single_rt
        if total_RT > B[hold_idx]*accpeted_rate:
            OBJ3 += total_RT - (B[hold_idx]*accpeted_rate)
        
print(OBJ3)

OBJ += w3 * OBJ3



