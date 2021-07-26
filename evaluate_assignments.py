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

BookingFile = "book/exp.csv"
AssignmentsFile = "/Users/takedakiyoshi/lab/kline/KLINE/out/exp_assignment.xlsx"
# AssignmentsFile = '/Users/takedakiyoshi/lab/kline/KLINE/ヒューリスティック/exp_2_5_assignment_86400.xlsx'
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


order_num = len(set(df["Order_ID"]))

V_ij = []
for i in I:
    tmp = []
    for j in range(order_num):
        tmp.append(0)
    V_ij.append(tmp)

OBJ = 0

for index, row in df.iterrows():
    # print(row)
    hold_idx = Hold_encode[Hold_encode["Hold"]==row["Hold_ID"]]["Index"].iloc[-1]
    order_idx = int(row["Order_ID"]-1)
    V_ij[hold_idx][order_idx] = row["Load_Units"]   


# for i  in range(len(V_ij)):
#     print(V_ij[i])

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
    if len(dport)>1:
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
total_orders = []
# 目的関数3
OBJ3 = 0
check_port = L[:-1] + D[:-1]

for hold_idx in range(len(V_ij)):
    orders = V_ij[hold_idx]
    accpeted_rate = Stress[hold_idx]
    order_array = []
    for port_num in range(len(T)):    
        tmp = []
        for i in range(len(orders)):
            tmp.append(0)
        order_array.append(tmp)
        #0で初期化
        
    for i in range(len(orders)):
        if orders[i]>0:
            lport = df[df["Order_ID"]==i+1]["LPORT"].iloc[-1]
            for j in range(int(lport),int(L[-1])+1):
                order_array[j][i] = orders[i]
 
            dport = df[df["Order_ID"]==i+1]["DPORT"].iloc[-1]
            for j in range(int(D[0]),int(dport)):
                order_array[j][i] = orders[i]

    total_orders.append(order_array)
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


#目的関数4 デッドスペース
OBJ4 = 0
penal4_k = 1000
check_port_dead_space = L[:-1]

deeper_holds = {}
for index,row in Ml_Back.iterrows():
    first_hold= int(row["Hold0"])
    tmp = []
    tmp_idx = 1
    deeper = []
    while tmp_idx <= 42 and str(row["Hold"+str(tmp_idx)])!='nan':
        deeper.append(int(row["Hold"+str(tmp_idx)]))
        tmp_idx += 1
    deeper_holds[first_hold] = deeper
accepted_filling_rate = pd.read_csv('data/accepted_filling_rate.csv') 

for port in check_port_dead_space:
    total_RT = []
    for hold_idx in range(len(V_ij)):
        loaded_rt = 0
        for i in range(len(total_orders[hold_idx][port])):
            load_unit = total_orders[hold_idx][port][i]
            if load_unit > 0: 
                single_rt = df[df["Order_ID"]==i+1]["RT"].iloc[-1]
                loaded_rt += single_rt*load_unit
        total_RT.append(loaded_rt)


    for hold_with_lamp in I_lamp:
        #許容充填率を超えているか確認
        if total_RT[hold_with_lamp] > B[hold_with_lamp]*accepted_filling_rate.at[hold_with_lamp,"Stress"]:
            deeper = deeper_holds[hold_with_lamp]
            for hold in deeper:
                if B[hold]-  total_RT[hold] >= 1:
                    OBJ4 += 1
print(OBJ4)

OBJ += w4 *penal4_k* OBJ4


# 目的関数5 残容量を入り口に寄せる
OBJ5 = 0
n_it = []
for hold_idx  in range(len(V_ij)):
    loaded_rt = 0
    for i in range(len(V_ij[hold_idx])):
        load_unit = V_ij[hold_idx][i]
        if load_unit > 0: 
            single_rt = df[df["Order_ID"]==i+1]["RT"].iloc[-1]
            loaded_rt += single_rt*load_unit
    n_it.append(B[hold_idx]-loaded_rt)
for i in range(len(n_it)):
    OBJ5 += n_it[i] * RT_benefit[i]

print(OBJ5)

OBJ -= w5 * OBJ5

print(OBJ)