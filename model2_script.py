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

def model2(FileName):

    def Read_booking(FileName):

        Booking_df = pd.read_csv(FileName)

        L = []  # 積み地の集合
        D = []  # 揚げ地の集合
        T = []  # 港の集合

        lport_name = list(Booking_df["PORT_L"].unique())
        dport_name = list(Booking_df["PORT_D"].unique())
        lport_name = lport_name[:-1]
        dport_name = dport_name[:-1]

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

        Booking = Booking_df.iloc[:, 0:Booking_df.columns.get_loc('PORT_L')]

        key2 = Booking.columns.get_loc('LPORT')
        key3 = Booking.columns.get_loc('DPORT')
        key4 = Booking.columns.get_loc('Units')
        key5 = Booking.columns.get_loc('RT')
        key6 = Booking.columns.get_loc('Weight')

        count1 = 0
        for col1 in L:
            for col2 in D:
                units = rt = weight = count2 = 0
                for k in range(len(Booking_df)):
                    if Booking_df.iloc[k, key2] == col1 and Booking_df.iloc[k, key3] == col2:
                        count2 += 1
                        units += Booking_df.iloc[k, key4]
                        rt += Booking_df.iloc[k, key4] * Booking_df.iloc[k, key5]
                        weight += Booking_df.iloc[k,
                                                key4] * Booking_df.iloc[k, key6]

                if count2 > 0:
                    Booking.iloc[count1, key2] = col1
                    Booking.iloc[count1, key3] = col2
                    Booking.iloc[count1, key4] = units
                    Booking.iloc[count1, key5] = rt
                    Booking.iloc[count1, key6] = int(weight / units)
                    count1 += 1
        print(Booking)

        for t in range(count1, len(Booking_df)):
            Booking = Booking.drop(Booking.index[count1])
        print(Booking)
        Booking["Index"] = 0
        for k in range(len(Booking)):
            Booking.iloc[k, Booking.columns.get_loc('Index')] = k

        A = list(Booking["RT"])
        J = list(Booking["Index"])
        U = list(Booking["Units"])
        G = list(Booking["Weight"])

        Port = Booking.iloc[:, key2:key3+1]

        # 注文をサイズ毎に分類
        J_small = []
        J_large = []
        for j in J:
            if A[j] <= 100:
                J_small.append(j)
            if 100 < A[j]:
                J_large.append(j)

        return T, L, D, J, U, A, G, J_small, J_large, Port, Check_port, Booking



    # 前処理
    # ==============================================================================================

    # ファイルロード
    BookingFile = FileName
    HoldFile = "data/hold.csv"
    MainLampFile = "data/mainlamp.csv"
    BackMainLampFile = "data/back_mainlamp.csv"
    AfrMainLampFile = "data/afr_mainlamp.csv"
    StressFile = "data/stress_mainlamp.csv"
    Gang2File = "data/gangnum_2.csv"
    Gang3File = "data/gangnum_3.csv"

    # 注文情報の読み込み
    # T, L, D, J, U, A, G, J_small,J_medium, J_large, Port, Check_port, Booking,diveded_dic = read_booking.Read_booking(BookingFile)
    T, L, D, J, U, A, G, J_small, J_large, Port, Check_port, Booking = Read_booking(BookingFile)
    print(J)
    # 船体情報の読み込み1
    I, B, I_pair, I_next, I_same, I_lamp, I_deck, RT_benefit, delta_s, min_s, max_s, delta_h, max_h, Hold_encode, Hold= read_hold.Read_hold(HoldFile)

    # 船体情報の読み込み2
    Ml_Load, Ml_Back, Ml_Afr, Stress, GANG2, GANG3 = read_other.Read_other(MainLampFile, BackMainLampFile, AfrMainLampFile, StressFile, Gang2File, Gang3File, Hold_encode)

    J_t_load = []  # J_t_load:港tで積む注文の集合
    J_t_keep = []  # J_t_keep:港tを通過する注文の集合
    J_t_dis = []  # J_t_dis:港tで降ろす注文の集合
    J_lk = []  # J_lk:J_t_load + J_t_keep
    J_ld = []  # J_ld:J_t_load + J_t_dis

    for t in T:

        J_load = []
        J_keep = []
        J_dis = []
        lk = []
        ld = []
        tmp_load = list(Port.iloc[:, 0])
        tmp_dis = list(Port.iloc[:, 1])
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


    # モデリング1(定数・変数の設定)
    # ==============================================================================================

    # Gurobiパラメータ設定
    GAP_SP = gp.Model()
    GAP_SP.setParam("TimeLimit", 3600)
    GAP_SP.setParam("MIPFocus", 1)
    GAP_SP.setParam("LPMethod", 1)
    GAP_SP.setParam("MIPGap",1)
    GAP_SP.printStats()

    # ハイパーパラメータ設定
    # 各目的関数の重み
    w1 = 1
    w2 = 1
    w3 = 1
    w4 = 1
    w5 = 0

    # 最小分割RT
    v_min = 10

    # 目的関数1のペナルティ
    penal1_z = 10

    # 目的関数2のペナルティ
    penal2_load = 1
    penal2_dis = 10

    # 目的関数3のペナルティ
    penal3_m = 10

    # 目的関数4のチェックポイント
    check_point = Check_port

    # 目的関数5のペナルティ
    penal5_k = 1000

    # 最適化変数
    # V_ij:注文jをホールドiにkRT割り当てる
    V_ij = {}
    for i in I:
        for j in J:
            V_ij[i, j] = GAP_SP.addVar(
                lb=0, ub=A[j], vtype=gp.GRB.CONTINUOUS, name=f"V_ij({i},{j})")

    # 目的関数1
    # X_ij:注文jをホールドiに割り当てるなら1、そうでなければ0
    X_ij = GAP_SP.addVars(I, J, vtype=gp.GRB.BINARY)

    # Y_keep_it:港tにおいてホールドiを通過する注文があるなら1、そうでなければ0
    Y_keep_it = GAP_SP.addVars(I, T, vtype=gp.GRB.BINARY)

    # Y_it1t2:ホールドiにおいてt1で積んでt2で降ろす注文があるなら1、そうでなければ0
    Y_it1t2 = GAP_SP.addVars(I, T, T, vtype=gp.GRB.BINARY)

    # Z_it1t2:ホールドiにおいてt2を通過する注文があるとき異なる乗せ港t1の数分のペナルティ
    Z_it1t2 = GAP_SP.addVars(I, L, D, vtype=gp.GRB.BINARY)

    OBJ1 = gp.quicksum(
        w1 * penal1_z * Z_it1t2[i, t1, t2] for i in I for t1 in L for t2 in D)

    # 目的関数2
    # Y_load_i1i2t:港tにおいてホールドペア(i1,i2)で積む注文があるなら1
    Y_load_i1i2t = GAP_SP.addVars(I, I, L, vtype=gp.GRB.BINARY)

    # Y_keep_i1i2t:港tにおいてホールドペア(i1,i2)を通過する注文があるなら1
    Y_keep_i1i2t = GAP_SP.addVars(I, I, T, vtype=gp.GRB.BINARY)

    # Y_dis_i1i2t:港tにおいてホールドペア(i1,i2)で揚げる注文があるなら1
    Y_dis_i1i2t = GAP_SP.addVars(I, I, D, vtype=gp.GRB.BINARY)

    # ホールドペア(i1,i2)においてtで注文を積む際に既にtで積んだ注文があるときのペナルティ
    Z1_i1i2t = GAP_SP.addVars(I, I, L, vtype=gp.GRB.BINARY)

    # ホールドペア(i1,i2)においてtで注文を揚げる際にtを通過する注文があるときのペナルティ
    Z2_i1i2t = GAP_SP.addVars(I, I, D, vtype=gp.GRB.BINARY)

    OBJ2_1 = gp.quicksum(
        penal2_load * Z1_i1i2t[i1, i2, t] for i1 in I for i2 in I for t in L)
    OBJ2_2 = gp.quicksum(
        penal2_dis * Z2_i1i2t[i1, i2, t] for i1 in I for i2 in I for t in D)
    OBJ2 = w2 * (OBJ2_1 + OBJ2_2)

    # 目的関数3
    # M_it:港tにおいてホールドiが作業効率充填率を超えたら1
    M_it = GAP_SP.addVars(I, T, vtype=gp.GRB.BINARY)

    # M_ijt:港tにおいてホールドiに自動車を積むまでに作業効率充填率を上回ったホールドに自動車を通すペナルティ
    M_ijt = GAP_SP.addVars(I, J, T, lb=0, vtype=gp.GRB.CONTINUOUS)

    OBJ3 = gp.quicksum(w3 * penal3_m * M_ijt[i, j, t]
                       for i in I for j in J for t in T)

    # 目的関数4
    # N_jt:チェックポイントにおけるホールドiの残容量
    N_it = GAP_SP.addVars(I, check_point, lb=0, vtype=gp.GRB.CONTINUOUS)

    OBJ4 = gp.quicksum(w4 * N_it[i, t] * RT_benefit[i]
                       for i in I for t in check_point)

    # 目的関数5
    # K1_it:lampで繋がっている次ホールドが充填率75%を上回ったら1、そうでなければ0
    K1_it = GAP_SP.addVars(I_lamp, L, vtype=gp.GRB.BINARY)

    # K2_it:ホールドiが1RT以上のスペースがあったら1、そうでなければ0
    K2_it = GAP_SP.addVars(I, L, lb=0, vtype=gp.GRB.BINARY)

    # K3_it:目的関数5のペナルティ
    K3_it = GAP_SP.addVars(I_lamp, L, lb=0, vtype=gp.GRB.CONTINUOUS)

    OBJ5 = gp.quicksum(w5 * penal5_k * K3_it[i, t] for i in I_lamp for t in L)

    # 目的関数の設計
    OBJ = OBJ1 + OBJ2 + OBJ3 - OBJ4 + OBJ5
    GAP_SP.setObjective(OBJ, gp.GRB.MINIMIZE)

    # モデリング2(制約)
    # ==============================================================================================

    # 基本制約
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # 割当てた注文がコンパートメント毎にリソースを超えない
    GAP_SP.addConstrs(gp.quicksum(V_ij[i, j] for j in J) <= B[i] for i in I)

    # 全注文内の自動車の台数を全て割り当てる
    GAP_SP.addConstrs(gp.quicksum(V_ij[i, j] for i in I) == A[j] for j in J)

    # VとXの制約
    GAP_SP.addConstrs(V_ij[i, j] / A[j] <= X_ij[i, j] for i in I for j in J)

    # 目的関数の制約
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # 目的関数1の制約
    for t in T:
        GAP_SP.addConstrs(X_ij[i, j] <= Y_keep_it[i, t]
                          for i in I for j in J_t_keep[t])

    for t in D:
        t2 = t
        for t1 in L:

            J_sub = []  # J_sub:t1で積んでt2で揚げる注文の集合
            for j in J_t_dis[t2]:
                if j in J_t_load[t1]:
                    J_sub.append(j)

            GAP_SP.addConstrs(X_ij[i, j] <= Y_it1t2[i, t1, t2]
                              for i in I for j in J_sub)
            GAP_SP.addConstrs(Z_it1t2[i, t1, t2] <=
                              Y_it1t2[i, t1, t2] for i in I)
            GAP_SP.addConstrs(Z_it1t2[i, t1, t2] <=
                              Y_keep_it[i, t2] for i in I)
            GAP_SP.addConstrs(
                Z_it1t2[i, t1, t2] >= Y_it1t2[i, t1, t2] + Y_keep_it[i, t2] - 1 for i in I)

    # 目的関数2の制約
    for t in T:
        for i1, i2 in I_pair:
            GAP_SP.addConstrs(X_ij[i1, j] + X_ij[i2, j] <=
                              2 * Y_keep_i1i2t[i1, i2, t] for j in J_t_keep[t])

            if t in L:
                GAP_SP.addConstrs(
                    X_ij[i1, j] + X_ij[i2, j] <= 2 * Y_load_i1i2t[i1, i2, t] for j in J_t_load[t])
                GAP_SP.addConstr(Z1_i1i2t[i1, i2, t]
                                 <= Y_load_i1i2t[i1, i2, t])
                GAP_SP.addConstr(Z1_i1i2t[i1, i2, t]
                                 <= Y_keep_i1i2t[i1, i2, t])
                GAP_SP.addConstr(
                    Z1_i1i2t[i1, i2, t] >= Y_load_i1i2t[i1, i2, t] + Y_keep_i1i2t[i1, i2, t] - 1)

            if t in D:
                GAP_SP.addConstrs(
                    X_ij[i1, j] + X_ij[i2, j] <= 2 * Y_dis_i1i2t[i1, i2, t] for j in J_t_dis[t])
                GAP_SP.addConstr(Z2_i1i2t[i1, i2, t] <= Y_dis_i1i2t[i1, i2, t])
                GAP_SP.addConstr(Z2_i1i2t[i1, i2, t]
                                 <= Y_keep_i1i2t[i1, i2, t])
                GAP_SP.addConstr(
                    Z2_i1i2t[i1, i2, t] >= Y_dis_i1i2t[i1, i2, t] + Y_keep_i1i2t[i1, i2, t] - 1)

    # 目的関数3の制約
    for t in T:
        GAP_SP.addConstrs(M_it[i, t] >= - Stress[i] + (gp.quicksum(V_ij[i, j]
                                                                   for j in J_t_keep[t]) / B[i]) for i in I)

        for i1 in I:

            I_primetmp = []
            for k in Ml_Load.iloc[i1, :]:
                I_primetmp.append(k)

            I_prime = [x for x in I_primetmp if str(x) != 'nan']
            I_prime.pop(0)

            GAP_SP.addConstrs(M_ijt[i1, j, t] >= gp.quicksum(
                M_it[i2, t] for i2 in I_prime) for j in J_ld[t])

    # 目的関数4の制約
    for t in check_point:
        GAP_SP.addConstrs(N_it[i, t] <= B[i] - gp.quicksum(V_ij[i, j]
                                                           for j in J_lk[t]) for i in I)

    # 目的関数5の制約
    GAP_SP.addConstrs(K1_it[i, t] >= - 0.75 + gp.quicksum(V_ij[i, j]
                                                          for j in J_lk[t]) / B[i] for i in I_lamp for t in L)
    GAP_SP.addConstrs(K2_it[i, t] >= 1 - (gp.quicksum(V_ij[i, j]
                                                      for j in J_lk[t]) + 1) / B[i] for i in I for t in L)

    for i in range(len(Ml_Back)):
        i1 = Ml_Back.iloc[i, 0]
        I_backtmp = []
        for k in Ml_Back.iloc[i, :]:
            I_backtmp.append(k)

        I_back_i1 = [x for x in I_backtmp if str(x) != 'nan']
        I_back_i1.pop(0)
        GAP_SP.addConstrs(K3_it[i1, t] >= len(
            I) * (K1_it[i1, t] - 1) + gp.quicksum(K2_it[i2, t] for i2 in I_back_i1) for t in L)

    # 特殊制約1(注文の分割制約)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # J_smallを分割しない
    GAP_SP.addConstrs(gp.quicksum(X_ij[i, j] for i in I) == 1 for j in J_small)

    # 10RT以下に分割しない
    GAP_SP.addConstrs(V_ij[i, j] + v_min * (1 - X_ij[i, j])
                      >= v_min for i in I for j in J)

    # 特殊制約2(移動経路制約)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # 港を通過する荷物が移動の邪魔をしない
    for t in T:
        for i1 in I:

            I_primetmp = []
            Frtmp = []

            for k in Ml_Load.iloc[i1, :]:
                I_primetmp.append(k)

            for k in Ml_Afr.iloc[i1, :]:
                Frtmp.append(k)

            I_prime = [int(x) for x in I_primetmp if str(x) != 'nan']
            Fr = [x for x in Frtmp if str(x) != 'nan']
            I_prime.pop(0)
            Fr.pop(0)

            N_prime = len(I_prime)
            for k in range(N_prime):
                i2 = int(I_prime[k])
                GAP_SP.addConstrs(gp.quicksum(
                    V_ij[i2, j1] for j1 in J_t_keep[t]) / B[i2] <= 1 + Fr[k] - X_ij[i1, j2] for j2 in J_ld[t])

    # 特殊制約3(船体重心の制約)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # 船の上下前後の配置バランスが閾値を超えない
    for t in T:

        # 荷物を全て降ろしたとき
        GAP_SP.addConstr(gp.quicksum(
            delta_h[i] * U[j] * G[j] * V_ij[i, j] / A[j] for j in J_t_keep[t] for i in I) <= max_h)
        GAP_SP.addConstr(gp.quicksum(
            delta_s[i] * U[j] * G[j] * V_ij[i, j] / A[j] for j in J_t_keep[t] for i in I) <= max_s)
        GAP_SP.addConstr(gp.quicksum(
            delta_s[i] * U[j] * G[j] * V_ij[i, j] / A[j] for j in J_t_keep[t] for i in I) >= min_s)

        # 荷物を全て載せたとき
        GAP_SP.addConstr(gp.quicksum(
            delta_h[i] * U[j] * G[j] * V_ij[i, j] / A[j] for j in J_lk[t] for i in I) <= max_h)
        GAP_SP.addConstr(gp.quicksum(
            delta_s[i] * U[j] * G[j] * V_ij[i, j] / A[j] for j in J_lk[t] for i in I) <= max_s)
        GAP_SP.addConstr(gp.quicksum(
            delta_s[i] * U[j] * G[j] * V_ij[i, j] / A[j] for j in J_lk[t] for i in I) >= min_s)

    # 特殊制約4(ギャングの制約)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    # 最適化計算
    # ==============================================================================================

    print("\n========================= Solve Assignment Problem =========================")
    GAP_SP.optimize()

    # 解の保存
    # ==============================================================================================

    # 目的関数の値
    val_opt = GAP_SP.ObjVal

    # OBJ1のペナルティ
    penal1 = 0
    for i in I:
        for t1 in L:
            for t2 in D:
                if Z_it1t2[i, t1, t2].X > 0:
                    penal1 = penal1 + Z_it1t2[i, t1, t2].X

    # OBJ2のペナルティ
    penal2_1 = penal2_2 = 0
    for i1 in I:
        for i2 in I:
            for t in L:
                if Z1_i1i2t[i1, i2, t].X > 0:
                    penal2_1 = penal2_1 + 1
                    # print(f"ホールド{i1},{i2}で積み地ペナルティ")
            for t in D:
                if Z2_i1i2t[i1, i2, t].X > 0:
                    penal2_2 = penal2_2 + 1
                    # print(f"ホールド{i1},{i2}で揚げ地ペナルティ")

    # OBJ3のペナルティ
    penal3 = 0
    for i in I:
        for j in J:
            for t in T:
                if M_ijt[i, j, t].X > 0:
                    penal3 = penal3 + M_ijt[i, j, t].X

    # OBJ4のペナルティ
    benefit4 = 0
    for t in check_point:
        for i in I:
            if N_it[i, t].X > 0:
                benefit4 = benefit4 + N_it[i, t].X

    # OBJ5のペナルティ
    penal5 = 0
    for t in L:
        for i in I_lamp:
            if K3_it[i, t].X > 0:
                penal5 = penal5 + K3_it[i, t].X

    # 解の書き込み
    answer = []
    assign = []
    for i in I:
        for j in J:
            if V_ij[i, j].X > 0:
                # assign_data[ホールド番号,積載RT,積み地,揚げ地]
                answer.append([i, j])
                assign.append([0, V_ij[i, j].X,  "L", "D"])
                


    # 残リソース
    I_left_data = Hold.iloc[:, 0:2]

    for k in range(len(assign)):

        key = Booking.columns.get_loc('Index')
        i_t = answer[k][0]
        j_t = int(Booking.iloc[answer[k][1], key])

        # hold_ID
        assign[k][0] = i_t



        # L_port
        assign[k][2] = Booking.iloc[j_t, Booking.columns.get_loc('LPORT')]

        # D_port
        assign[k][3] = Booking.iloc[j_t, Booking.columns.get_loc('DPORT')]


    
    csv = []
    for k in range(len(assign)):
        csv.append(assign[k])

    c_list = []
    c_list.append("Hold_ID")
    c_list.append("Load_RT")
    c_list.append("LPORT")
    c_list.append("DPORT")
    assign_data = pd.DataFrame(csv, columns=c_list)

    return assign_data
    
    # # 解を全て格納する配列
    # relaxed_all_assignment = []
    
    # # 解の書き込み
    # answer = []
    # assign = []
    # for i in I:
    #     tmp_assignment = []
    #     for j in J:
    #         tmp_assignment.append(V_ij[i, j].X)
    #     relaxed_all_assignment.append(tmp_assignment) 

    # initial_assingment_T = np.array(relaxed_all_assignment).T
    # return initial_assingment_T



