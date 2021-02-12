import warnings
import openpyxl
import numpy as np
import pandas as pd
import gurobipy as gp
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
import sys

warnings.filterwarnings("ignore")

args = sys.argv

def Read_booking(BookingFileName):
    
    Booking_df = pd.read_csv(BookingFileName)
    
    L = [] #積み地の集合
    D = [] #揚げ地の集合
    T = [] #港の集合
    
    lport_name = [x for x in list(Booking_df["PORT_L"].unique()) if not pd.isnull(x)]
    dport_name = [x for x in list(Booking_df["PORT_D"].unique()) if not pd.isnull(x)]

    for t in range(len(lport_name)):
        L.append(t)
    
    for t in range(len(dport_name)):
        D.append(t + max(L) + 1)
        
    T = L + D

    Check_port = []
    key1 = Booking_df.columns.get_loc('CPORT')
    for j in range(len(Booking_df)):
        count = 0
        if str(Booking_df.iloc[j,key1]) != 'nan':
            for p in lport_name:
                if p == str(Booking_df.iloc[j,key1]):
                    Check_port.append(count)
                else:
                    count = count + 1
    
    #港番号のエンコード
    for k in range(len(lport_name)):
        Booking_df = Booking_df.replace(lport_name[k], L[k])
    
    for k in range(len(dport_name)):
        Booking_df = Booking_df.replace(dport_name[k], D[k])
    
    #巨大な注文の分割
    divided_j = []
    divide_dic = []
    divide_df = Booking_df.iloc[0,:]

    for j in range(len(Booking_df)):
        tmp = Booking_df.iloc[j, Booking_df.columns.get_loc('Units')]
        if 500 < tmp and tmp <= 1000:
            if tmp % 2 == 0:
                u_num = [int(tmp / 2), int(tmp / 2)]
            if tmp % 2 == 1:
                u_num = [int(tmp / 2) + 1, int(tmp / 2)]
            concat1 = concat2 = Booking_df.iloc[j,:]
            concat1["Units"] = u_num[0]
            concat2["Units"] = u_num[1]
            divide_df = pd.concat([divide_df, concat1, concat2], axis = 1)
            divided_j.append(j)
            divide_dic.append([j,tmp])
       
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
    VehicleHeight = list(Booking["Height"])
    
    key2 = Booking.columns.get_loc('LPORT')
    key3 = Booking.columns.get_loc('DPORT')
    Port = Booking.iloc[:,key2:key3+1]
    
    #注文をサイズ毎に分類
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
 
    return T,L,D,J,U,A,G,J_small,J_medium,J_large,Port,Check_port,Booking,divide_dic, VehicleHeight

def Read_hold(HoldFileName):
    
    Hold_df = pd.read_csv(HoldFileName)
    
    B = list(Hold_df["Resourse"])
    RT_benefit = list(Hold_df["RT_benefit"])
    delta_s = list(Hold_df["Weight_s"])
    delta_h = list(Hold_df["Weight_h1"] * Hold_df["Weight_h2"])
    min_s = Hold_df.iloc[0,Hold_df.columns.get_loc('Min_s')]
    max_s = Hold_df.iloc[0,Hold_df.columns.get_loc('Max_s')]
    max_h = Hold_df.iloc[0,Hold_df.columns.get_loc('Max_h')]
    height = list(Hold_df["Height"])
    

    
    list_drop_cols = ['Resourse','RT_benefit','Weight_s','Weight_h1','Weight_h2','Min_s','Max_s','Max_h','Height']
    
    #ホールド番号のエンコード
    Hold_encode = Hold_df.iloc[:,0:2]
    le = LabelEncoder()
    Hold_encode["Resourse"] = le.fit_transform(Hold_df['Hold'].values)
    Hold_encode = Hold_encode.rename(columns = {'Resourse':'Index'})
    Hold_data = Hold_df.drop(list_drop_cols,axis=1)
    
    for i in range(len(Hold_encode)):
        Hold_data = Hold_data.replace(Hold_encode.iloc[i,0], Hold_encode.iloc[i,1])
    
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
        if str(Hold_data.iloc[i,key1]) != 'nan':
            I_pair.append([int(Hold_data.iloc[i,key1]), int(Hold_data.iloc[i,key1+1])])
        if str(Hold_data.iloc[i,key2]) != 'nan':
            I_next.append([int(Hold_data.iloc[i,key2]), int(Hold_data.iloc[i,key2+1])])
        if str(Hold_data.iloc[i,key3]) != 'nan':
            I_same.append([int(Hold_data.iloc[i,key3]), int(Hold_data.iloc[i,key3+1])])
        if str(Hold_data.iloc[i,key4]) != 'nan':
            I_lamp.append(int(Hold_data.iloc[i,key4]))
    
    last = len(Hold_data.T)
    for i in range(key5,last):
        append_list = []
        count = 0
        while str(Hold_data.iloc[count,i]) != 'nan':
            append_list.append(int(Hold_data.iloc[count,i]))
            count = count + 1
        I_deck.append(append_list)
       
    return I,B,I_pair,I_next,I_same,I_lamp,I_deck,RT_benefit,delta_s,min_s,max_s,delta_h,max_h,Hold_encode,Hold_df,height

def Read_other(FileName1,FileName2,FileName3,FileName4,FileName5,FileName6,Hold_encode):
    
    Ml_Load = pd.read_csv(FileName1)
    Ml_Back = pd.read_csv(FileName2)
    Ml_Afr = pd.read_csv(FileName3)
    Stress = pd.read_csv(FileName4)
    g_2 = pd.read_csv(FileName5)
    g_3 = pd.read_csv(FileName6)
    
    for i in range(len(Hold_encode)):
        g_2 = g_2.replace(Hold_encode.iloc[i,0], Hold_encode.iloc[i,1])
        g_3 = g_3.replace(Hold_encode.iloc[i,0], Hold_encode.iloc[i,1])
        Ml_Load = Ml_Load.replace(Hold_encode.iloc[i,0], Hold_encode.iloc[i,1])
        Ml_Back = Ml_Back.replace(Hold_encode.iloc[i,0], Hold_encode.iloc[i,1])
        Ml_Afr = Ml_Afr.replace(Hold_encode.iloc[i,0], Hold_encode.iloc[i,1])
    
    Stress = list(Stress.iloc[:,1])
    
    GANG2 = []
    for n in range(2):
        add = []
        for k in range(len(g_2.iloc[:,n])):
            if str(g_2.iloc[k,n]) != 'nan':
                add.append(int(g_2.iloc[k,n]))
        GANG2.append(add)
            
    GANG3 = []
    for n in range(3):
        add = []
        for k in range(len(g_3.iloc[:,n])):
            if str(g_3.iloc[k,n]) != 'nan':
                add.append(int(g_3.iloc[k,n]))
        GANG3.append(add)
        
    return Ml_Load,Ml_Back,Ml_Afr,Stress,GANG2,GANG3

def main():

    #前処理
    #==============================================================================================
    
    #ファイルロード
    # BookingFile = "book/exp_height.csv"
    BookingFile = args[1]
    HoldFile = "revised_data/hold.csv"
    MainLampFile = "revised_data/mainlamp.csv"
    BackMainLampFile = "revised_data/back_mainlamp.csv"
    AfrMainLampFile = "revised_data/afr_mainlamp.csv"
    StressFile = "revised_data/stress_mainlamp.csv"
    Gang2File = "data/gangnum_2.csv"
    Gang3File = "data/gangnum_3.csv"

    print("File:" + BookingFile)
    
    #注文情報の読み込み
    T,L,D,J,U,A,G,J_small,J_medium,J_large,Port,Check_port,Booking,divide_dic, VehicleHeight = Read_booking(BookingFile)
    
    #船体情報の読み込み1
    I,B,I_pair,I_next,I_same,I_lamp,I_deck,RT_benefit,delta_s,min_s,max_s,delta_h,max_h,Hold_encode,Hold, HoldHeight = Read_hold(HoldFile)
    
    #船体情報の読み込み2
    Ml_Load,Ml_Back,Ml_Afr,Stress,GANG2,GANG3 = Read_other(MainLampFile,BackMainLampFile,AfrMainLampFile,StressFile,Gang2File,Gang3File,Hold_encode)
    
    
    
    J_t_load = [] #J_t_load:港tで積む注文の集合
    J_t_keep = [] #J_t_keep:港tを通過する注文の集合
    J_t_dis  = [] #J_t_dis:港tで降ろす注文の集合
    J_lk = [] #J_lk:J_t_load + J_t_keep
    J_ld = [] #J_ld:J_t_load + J_t_dis
    
    for t in T:
        
        J_load = []
        J_keep = []
        J_dis  = []
        lk = []
        ld = []
        tmp_load = list(Port.iloc[:,0])
        tmp_dis = list(Port.iloc[:,1])
        N = len(J)
        
        k = 0
        for i in L:
            if k < i:
                k = i
                
        count = 0
        for t_l in tmp_load:
            if t == t_l:
                J_load.append(count)
                lk.append(count)
                ld.append(count)
            count = count + 1
        
        count = 0
        for t_d in tmp_dis:
            if t == t_d:
                J_dis.append(count)
                ld.append(count)
            count = count + 1
        
        for t_k in range(N):
            if t > tmp_load[t_k] and t < tmp_dis[t_k]:
                J_keep.append(J[t_k])
                lk.append(t_k)
        
        J_t_load.append(J_load)
        J_t_keep.append(J_keep)
        J_t_dis.append(J_dis)
        J_lk.append(lk)
        J_ld.append(ld)
    
    gang_num = np.zeros(len(L)) #各港のギャング数
    J_N = 0
    for t in L:
        for j in J_t_load[t]:
            J_N = J_N + U[j]
        if J_N <= 500:
            gang_num[t] = 1
        if J_N > 500 and J_N <= 1000:
            gang_num[t] = 2
        if J_N > 1000:
            gang_num[t] = 3
    
    """     
    J_large_divide = [] #大きい注文の分割サイズ指定
    for j in J_large:
        tmp = []
        for t in L:
            if j in J_t_load[t]:
                p = t
                if U[j] % gang_num[p] == 0:
                    for k in range(int(gang_num[p])):
                        tmp.append(int(U[j] / gang_num[p]))
                    J_large_divide.append(tmp)  
                if U[j] % gang_num[p] == 1:
                    num = int(U[j] / gang_num[p])   
                    tmp_num = U[j]
                    while tmp_num > num + 1:
                        tmp.append(num)
                        tmp_num = tmp_num - num
                    tmp.append(tmp_num)
                    J_large_divide.append(tmp)
                if U[j] % gang_num[p] == 2:
                   num = int(U[j] / gang_num[p]) + 1  
                   tmp_num = U[j]
                   while tmp_num > num:
                       tmp.append(num)
                       tmp_num = tmp_num - num
                   tmp.append(tmp_num)
                   J_large_divide.append(tmp)
    """
    
    
    
    #モデリング1(定数・変数の設定)
    #==============================================================================================
    
    #Gurobiパラメータ設定
    GAP_SP = gp.Model()
    GAP_SP.setParam("TimeLimit", 86400)
    GAP_SP.setParam("MIPFocus", 1)
    GAP_SP.setParam("LPMethod", 1)
    GAP_SP.printStats()
    
    #ハイパーパラメータ設定
    #各目的関数の重み
    w1 = 1
    w2 = 1
    w3 = 1
    w4 = 1
    w5 = 1
    
    #目的関数1のペナルティ
    penal1_z = 10
    
    #目的関数2のペナルティ
    penal2_load = 1
    penal2_dis = 10
    
    #目的関数4のチェックポイント
    check_point = Check_port
 
    #目的関数5のペナルティ
    penal5_k = 1000
    
    #最適化変数
    #V_ij:注文jをホールドiにk(kはUnit数)台割り当てる
    V_ij = {}
    for i in I:
        for j in J:
            V_ij[i,j] = GAP_SP.addVar(lb = 0, ub = U[j], vtype = gp.GRB.INTEGER, name = f"V_ij({i},{j})")
    
    #目的関数1
    #X_ij:注文jをホールドiに割り当てるなら1、そうでなければ0
    X_ij = GAP_SP.addVars(I,J, vtype = gp.GRB.BINARY)
    
    #Y_keep_it:港tにおいてホールドiを通過する注文があるなら1、そうでなければ0
    Y_keep_it = GAP_SP.addVars(I,T, vtype = gp.GRB.BINARY)
    
    #Y_it1t2:ホールドiにおいてt1で積んでt2で降ろす注文があるなら1、そうでなければ0
    Y_it1t2 = GAP_SP.addVars(I,T,T, vtype = gp.GRB.BINARY)
    
    #Z_it1t2:ホールドiにおいてt2を通過する注文があるとき異なる乗せ港t1の数分のペナルティ
    Z_it1t2 = GAP_SP.addVars(I,L,D, vtype = gp.GRB.BINARY)
    
    OBJ1 = gp.quicksum(w1 * penal1_z * Z_it1t2[i,t1,t2] for i in I for t1 in L for t2 in D)
    
    #目的関数2
    #Y_load_i1i2t:港tにおいてホールドペア(i1,i2)で積む注文があるなら1
    Y_load_i1i2t = GAP_SP.addVars(I,I,L, vtype = gp.GRB.BINARY)
    
    #Y_keep_i1i2t:港tにおいてホールドペア(i1,i2)を通過する注文があるなら1
    Y_keep_i1i2t = GAP_SP.addVars(I,I,T, vtype = gp.GRB.BINARY)
    
    #Y_dis_i1i2t:港tにおいてホールドペア(i1,i2)で揚げる注文があるなら1
    Y_dis_i1i2t = GAP_SP.addVars(I,I,D, vtype = gp.GRB.BINARY)

    #ホールドペア(i1,i2)においてtで注文を積む際に既にtで積んだ注文があるときのペナルティ
    Z1_i1i2t = GAP_SP.addVars(I,I,L, vtype = gp.GRB.BINARY)
    
    #ホールドペア(i1,i2)においてtで注文を揚げる際にtを通過する注文があるときのペナルティ
    Z2_i1i2t = GAP_SP.addVars(I,I,D, vtype = gp.GRB.BINARY)
    
    OBJ2_1 = gp.quicksum(penal2_load * Z1_i1i2t[i1,i2,t] for i1 in I for i2 in I for t in L)
    OBJ2_2 = gp.quicksum(penal2_dis * Z2_i1i2t[i1,i2,t] for i1 in I for i2 in I for t in D)
    OBJ2 = w2 * (OBJ2_1 + OBJ2_2)
    
    #目的関数3
    #M_it:港tにおいてホールドiが作業効率充填率を超えたら1
    M_it = GAP_SP.addVars(I,T, vtype = gp.GRB.BINARY)
    
    #M_ijt:港tにおいてホールドiに自動車を積むまでに作業効率充填率を上回ったホールドに自動車を通すペナルティ
    M_ijt = GAP_SP.addVars(I,J,T, lb = 0, vtype = gp.GRB.CONTINUOUS)
    
    OBJ3 = gp.quicksum(w3 * M_ijt[i,j,t] for i in I for j in J for t in T)  
    
    #目的関数4
    #N_jt:チェックポイントにおけるホールドiの残容量
    N_it = GAP_SP.addVars(I,check_point, lb = 0, vtype = gp.GRB.CONTINUOUS)
    
    OBJ4 = gp.quicksum(w4 * N_it[i,t] * RT_benefit[i] for i in I for t in check_point) 
    
    #目的関数5
    #K1_it:lampで繋がっている次ホールドが充填率75%を上回ったら1、そうでなければ0
    K1_it = GAP_SP.addVars(I_lamp,L, vtype = gp.GRB.BINARY)
    
    #K2_it:ホールドiが1RT以上のスペースがあったら1、そうでなければ0
    K2_it = GAP_SP.addVars(I,L, lb = 0, vtype = gp.GRB.BINARY)
    
    #K3_it:目的関数5のペナルティ
    K3_it = GAP_SP.addVars(I_lamp,L, lb = 0, vtype = gp.GRB.CONTINUOUS)
    
    OBJ5 = gp.quicksum(w5 * penal5_k * K3_it[i,t] for i in I_lamp for t in L)  
    
    #目的関数の設計
    OBJ = OBJ1 + OBJ2 + OBJ3 - OBJ4 + OBJ5
    GAP_SP.setObjective(OBJ, gp.GRB.MINIMIZE)

    #モデリング2(制約)
    #==============================================================================================
    
    #基本制約
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

    #高さ制約
    #ホールドの高さ
    HoldHeightVar = []
    for i in range(len(HoldHeight)):
        HoldHeightVar.append(GAP_SP.addVar(
            lb=0, vtype=gp.GRB.CONTINUOUS))
    #車の高さ
    VehicleHeightVar = []
    for i in range(len(VehicleHeight)):
        VehicleHeightVar.append(GAP_SP.addVar(
            lb=0, vtype=gp.GRB.CONTINUOUS))

    #制約式
    for i in range(len(HoldHeight)):
        for j in range(len(VehicleHeight)):
            GAP_SP.addConstr(VehicleHeightVar[i]*X_ij[i, j] <= HoldHeightVar[i])
    
    #割当てた注文がコンパートメント毎にリソースを超えない
    GAP_SP.addConstrs(gp.quicksum(V_ij[i,j] * A[j] for j in J) <= B[i] for i in I)
    
    #全注文内の自動車の台数を全て割り当てる
    GAP_SP.addConstrs(gp.quicksum(V_ij[i,j] * X_ij[i,j] for i in I) == U[j] for j in J)
    
    #VとXの制約
    GAP_SP.addConstrs(V_ij[i,j] / U[j] <= X_ij[i,j] for i in I for j in J)
    
    #目的関数の制約
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    
    #目的関数1の制約
    for t in T:
        GAP_SP.addConstrs(X_ij[i,j] <= Y_keep_it[i,t] for i in I for j in J_t_keep[t])
    
    for t in D:
        t2 = t
        for t1 in L:
        
            J_sub = [] #J_sub:t1で積んでt2で揚げる注文の集合
            for j in J_t_dis[t2]:
                if j in J_t_load[t1]:
                    J_sub.append(j)
            
            GAP_SP.addConstrs(X_ij[i,j] <= Y_it1t2[i,t1,t2] for i in I for j in J_sub)
            GAP_SP.addConstrs(Z_it1t2[i,t1,t2] <= Y_it1t2[i,t1,t2] for i in I)
            GAP_SP.addConstrs(Z_it1t2[i,t1,t2] <= Y_keep_it[i,t2] for i in I)
            GAP_SP.addConstrs(Z_it1t2[i,t1,t2] >= Y_it1t2[i,t1,t2] + Y_keep_it[i,t2] - 1 for i in I)
           
    #目的関数2の制約
    for t in T:
        for i1,i2 in I_pair: 
            GAP_SP.addConstrs(X_ij[i1,j] + X_ij[i2,j] <= 2 * Y_keep_i1i2t[i1,i2,t] for j in J_t_keep[t])
            
            if t in L:
                GAP_SP.addConstrs(X_ij[i1,j] + X_ij[i2,j] <= 2 * Y_load_i1i2t[i1,i2,t] for j in J_t_load[t])
                GAP_SP.addConstr(Z1_i1i2t[i1,i2,t] <= Y_load_i1i2t[i1,i2,t])
                GAP_SP.addConstr(Z1_i1i2t[i1,i2,t] <= Y_keep_i1i2t[i1,i2,t])
                GAP_SP.addConstr(Z1_i1i2t[i1,i2,t] >= Y_load_i1i2t[i1,i2,t] + Y_keep_i1i2t[i1,i2,t] - 1)
            
            if t in D:
                GAP_SP.addConstrs(X_ij[i1,j] + X_ij[i2,j] <= 2 * Y_dis_i1i2t[i1,i2,t] for j in J_t_dis[t])
                GAP_SP.addConstr(Z2_i1i2t[i1,i2,t] <= Y_dis_i1i2t[i1,i2,t])
                GAP_SP.addConstr(Z2_i1i2t[i1,i2,t] <= Y_keep_i1i2t[i1,i2,t])
                GAP_SP.addConstr(Z2_i1i2t[i1,i2,t] >= Y_dis_i1i2t[i1,i2,t] + Y_keep_i1i2t[i1,i2,t] - 1)
    
    #目的関数3の制約
    for t in T:
        GAP_SP.addConstrs(M_it[i,t] >= - Stress[i] + (gp.quicksum(V_ij[i,j] * A[j] for j in J_t_keep[t]) / B[i]) for i in I)
        
        for i1 in I:
            
            I_primetmp = []
            for k in  Ml_Load.iloc[i1,:]:
                I_primetmp.append(k)
            
            I_prime = [x for x in I_primetmp if str(x) != 'nan']
            I_prime.pop(0)
            
            GAP_SP.addConstrs(M_ijt[i1,j,t] >= V_ij[i1,j] * gp.quicksum(M_it[i2,t] for i2 in I_prime) for j in J_ld[t])
    
    #目的関数4の制約
    for t in check_point:
        GAP_SP.addConstrs(N_it[i,t] <= B[i] - gp.quicksum(V_ij[i,j] * A[j] for j in J_lk[t]) for i in I)
    
    #目的関数5の制約
    GAP_SP.addConstrs(K1_it[i,t] >= - 0.75 + gp.quicksum(V_ij[i,j] * A[j] for j in J_lk[t]) / B[i] for i in I_lamp for t in L)
    GAP_SP.addConstrs(K2_it[i,t] >= 1 - (gp.quicksum(V_ij[i,j] * A[j] for j in J_lk[t]) + 1) / B[i] for i in I for t in L)
           
    for i in range(len(Ml_Back)):
        i1 = Ml_Back.iloc[i,0]
        I_backtmp = []
        for k in  Ml_Back.iloc[i,:]:
            I_backtmp.append(k)
                    
        I_back_i1 = [x for x in I_backtmp if str(x) != 'nan']
        I_back_i1.pop(0)
        GAP_SP.addConstrs(K3_it[i1,t] >= len(I) * (K1_it[i1,t]- 1) + gp.quicksum(K2_it[i2,t] for i2 in I_back_i1) for t in L)
        
    #特殊制約1(注文の分割制約)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    
    #J_small
    #分割しない
    # GAP_SP.addConstrs(gp.quicksum(X_ij[i,j] for i in I) == 1 for j in J_small)
    
    #J_medium
    # GAP_SP.addConstrs(gp.quicksum(X_ij[i,j] for i in I) == 1 for j in J_medium)
    
    #J_large
    # for k1 in range(len(I_deck)):
    #     for k2 in range(k1):
    #         GAP_SP.addConstrs(gp.quicksum(X_ij[i1,j] for i1 in I_deck[k1]) * gp.quicksum(X_ij[i2,j] for i2 in I_deck[k2]) <= 0 for j in J_large)
       
    #特殊制約2(移動経路制約)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    #港を通過する荷物が移動の邪魔をしない
    for t in T:
        for i1 in I:
            
            I_primetmp = []
            Frtmp = []
            
            for k in  Ml_Load.iloc[i1,:]:
                I_primetmp.append(k)
            
            for k in  Ml_Afr.iloc[i1,:]:
                Frtmp.append(k)
        
            I_prime = [int(x) for x in I_primetmp if str(x) != 'nan']
            Fr = [x for x in Frtmp if str(x) != 'nan']
            I_prime.pop(0)
            Fr.pop(0)
            
            N_prime = len(I_prime)
            for k in range(N_prime):
                i2 = int(I_prime[k])
                GAP_SP.addConstrs(gp.quicksum(V_ij[i2,j1] * A[j1] for j1 in J_t_keep[t]) / B[i2]  <= 1 + Fr[k] - X_ij[i1,j2] for j2 in J_ld[t])
             
    #特殊制約3(船体重心の制約)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    #船の上下前後の配置バランスが閾値を超えない
    for t in T:

        #荷物を全て降ろしたとき
        GAP_SP.addConstr(gp.quicksum(delta_h[i] * G[j] * V_ij[i,j] for j in J_t_keep[t] for i in I) <= max_h)
        GAP_SP.addConstr(gp.quicksum(delta_s[i] * G[j] * V_ij[i,j] for j in J_t_keep[t] for i in I) <= max_s)
        GAP_SP.addConstr(gp.quicksum(delta_s[i] * G[j] * V_ij[i,j] for j in J_t_keep[t] for i in I) >= min_s)
        
        #荷物を全て載せたとき
        GAP_SP.addConstr(gp.quicksum(delta_h[i] * G[j] * V_ij[i,j] for j in J_lk[t] for i in I) <= max_h)
        GAP_SP.addConstr(gp.quicksum(delta_s[i] * G[j] * V_ij[i,j] for j in J_lk[t] for i in I) <= max_s)
        GAP_SP.addConstr(gp.quicksum(delta_s[i] * G[j] * V_ij[i,j] for j in J_lk[t] for i in I) >= min_s)
    
    #特殊制約4(ギャングの制約)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    """
    count = 0
    for t in L:
       gang = gang_num[count] 
       
       if gang == 2:
           gang_low = GANG2[0]
           gang_high = GANG2[1]
            
           #ギャングの担当領域毎に均等に注文を分ける
           GAP_SP.addConstr(gp.quicksum(V_ij[i1,j] for i1 in gang_low for j in J_t_load[t]) - gp.quicksum(V_ij[i2,j] for i2 in gang_high for j in J_t_load[t]) <= 100)
           GAP_SP.addConstr(gp.quicksum(V_ij[i1,j] for i1 in gang_low for j in J_t_load[t]) - gp.quicksum(V_ij[i2,j] for i2 in gang_high for j in J_t_load[t]) >= -100)
              
       if gang == 3:
           gang_low = GANG3[0]
           gang_mid = GANG3[1]
           gang_high = GANG3[2]
           
           #ギャングの担当領域毎に均等に注文を分ける
           GAP_SP.addConstr(gp.quicksum(V_ij[i1,j] for i1 in gang_low for j in J_t_load[t]) - gp.quicksum(V_ij[i2,j] for i2 in gang_mid for j in J_t_load[t]) <= 100)
           GAP_SP.addConstr(gp.quicksum(V_ij[i1,j] for i1 in gang_low for j in J_t_load[t]) - gp.quicksum(V_ij[i2,j] for i2 in gang_mid for j in J_t_load[t]) >= -100)
           GAP_SP.addConstr(gp.quicksum(V_ij[i1,j] for i1 in gang_low for j in J_t_load[t]) - gp.quicksum(V_ij[i2,j] for i2 in gang_high for j in J_t_load[t]) <= 100)
           GAP_SP.addConstr(gp.quicksum(V_ij[i1,j] for i1 in gang_low for j in J_t_load[t]) - gp.quicksum(V_ij[i2,j] for i2 in gang_high for j in J_t_load[t]) >= -100)
           GAP_SP.addConstr(gp.quicksum(V_ij[i1,j] for i1 in gang_mid for j in J_t_load[t]) - gp.quicksum(V_ij[i2,j] for i2 in gang_high for j in J_t_load[t]) <= 100)
           GAP_SP.addConstr(gp.quicksum(V_ij[i1,j] for i1 in gang_mid for j in J_t_load[t]) - gp.quicksum(V_ij[i2,j] for i2 in gang_high for j in J_t_load[t]) >= -100)
    """
    
    #最適化計算
    #==============================================================================================
   
    print("\n========================= Solve Assignment Problem =========================")
    GAP_SP.optimize()
    
    #解の保存
    #==============================================================================================
    
    #目的関数の値
    val_opt = GAP_SP.ObjVal
    
    #ペナルティ計算
    print("-" * 65)
    print("penalty count => ")
    
    #OBJ1のペナルティ
    penal1 = 0
    for i in I:
        for t1 in L:
            for t2 in D:
                if Z_it1t2[i,t1,t2].X > 0:
                    penal1 = penal1 + Z_it1t2[i,t1,t2].X
                    
    #OBJ2のペナルティ
    penal2_1 = penal2_2 = 0
    for i1 in I:
        for i2 in I:
            for t in L:
                if Z1_i1i2t[i1,i2,t].X > 0:
                    penal2_1 = penal2_1 + 1
                    #print(f"ホールド{i1},{i2}で積み地ペナルティ")
            for t in D:
                if Z2_i1i2t[i1,i2,t].X > 0:
                   penal2_2 = penal2_2 + 1
                   #print(f"ホールド{i1},{i2}で揚げ地ペナルティ")
               
    #OBJ3のペナルティ
    penal3 = 0
    for i in I:
        for j in J:
            for t in T:
                if M_ijt[i,j,t].X > 0:
                    penal3 = penal3 + M_ijt[i,j,t].X
    
    #OBJ4のペナルティ
    benefit4 = 0
    for t in check_point:
        for i in I:
            if N_it[i,t].X > 0:
                benefit4 = benefit4 + N_it[i,t].X
    
    #OBJ5のペナルティ
    penal5 = 0
    for t in L:
        for i in I_lamp:
            if K3_it[i,t].X > 0:
                penal5 = penal5 + K3_it[i,t].X
                
    #解の書き込み
    answer = []
    assign = []            
    for i in I:              
        for j in J:
            if V_ij[i,j].X > 0:
                #assign_data[GAPホールド番号、GAP注文番号、積む台数,ホールド番号、注文番号、ユニット数、RT、積み港、降ろし港、資源要求量]
                answer.append([i,j])
                assign.append([0,0,V_ij[i,j].X,0,0,"L","D",0])
    
    print("")
    print("[weight]")
    print(f"Object1's weight : {w1}")
    print(f"Object2's weight : {w2}")
    print(f"Object3's weight : {w3}")
    print(f"Object4's weight : {w4}")
    print(f"Object5's weight : {w5}")
    
    print("")
    print("[penalty & benefit]")
    print(f"Different discharge port in one hold                  : {penal1_z} × {penal1}")  
    print(f"Different loading port in pair hold                   : {penal2_load} × {penal2_1}")
    print(f"Different discharge port in pair hold                 : {penal2_dis} × {penal2_2}")
    print(f"The number of cars passed holds that exceed threshold : {penal3}")
    print(f"The benefit remaining RT of hold near the entrance    : {benefit4}")
    print(f"The penalty of total dead space                       : {penal5_k} × {penal5}")
    print("")             
    print(f" => Total penalty is {val_opt}")
    print("-" * 65)  
    
    #残リソース 
    I_left_data = Hold.iloc[:,0:2]
    
    for k in range(len(assign)):
        
        key = Booking.columns.get_loc('Index')
        i_t = answer[k][0]
        j_t = Booking.iloc[answer[k][1], key]
        
        #hold_ID
        assign[k][0] = Hold.iloc[i_t,0]
        
        #order_ID
        assign[k][1] = Booking.iloc[j_t,Booking.columns.get_loc('Order_num')]
        
        #Units(original booking)
        assign[k][3] = Booking.iloc[j_t,Booking.columns.get_loc('Units')]
        for j in range(len(divide_dic)):
            if assign[k][1] - 1 == divide_dic[j][0]:
                assign[k][3] = divide_dic[j][1]
        
        #RT
        assign[k][4] = Booking.iloc[j_t,Booking.columns.get_loc('RT')]
        
        #L_port
        assign[k][5] = Booking.iloc[j_t,Booking.columns.get_loc('LPORT')]
        
        #D_port
        assign[k][6] = Booking.iloc[j_t,Booking.columns.get_loc('DPORT')]
        
        #Cost
        assign[k][7] = assign[k][2] * assign[k][4]
        
        #残リソース計算
        I_left_data.iloc[answer[k][0], 1] = I_left_data.iloc[answer[k][0], 1] - assign[k][7]
        if I_left_data.iloc[answer[k][0], 1] < 0.1:
            I_left_data.iloc[answer[k][0], 1] = 0

    c_list = []
    c_list.append("Hold_ID")
    c_list.append("Order_ID")
    c_list.append("Load_Units")
    c_list.append("Units")
    c_list.append("RT")
    c_list.append("LPORT")
    c_list.append("DPORT")
    c_list.append("Cost")
    booking_name = BookingFile.split(".")[0]
    assignment_result_name="result/height_"+booking_name+"_assignment.xlsx"
    leftRT_result_name="result/"+booking_name+"_leftRT.xlsx"
    assign_data = pd.DataFrame(assign, columns=c_list)
    assign_data.to_excel(assignment_result_name, index=False, columns=c_list)
    I_left_data.to_excel(leftRT_result_name, index=False)
    
if __name__ == "__main__":
    main()

