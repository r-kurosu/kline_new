import warnings
import numpy as np
import pandas as pd
import gurobipy as gp
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
import read_booking
import read_hold
import read_other
import operation
import random
import sys
args = sys.argv
import itertools
import copy
import datetime
import model2_script
warnings.filterwarnings("ignore")


def main():
    # BookingFile = "book/exp.csv"
    BookingFile = args[1]
    HoldFile = "data/hold.csv"
    MainLampFile = "data/mainlamp.csv"
    BackMainLampFile = "data/back_mainlamp.csv"
    AfrMainLampFile = "data/afr_mainlamp.csv"
    AfrFile = "data/accepted_filling_rate.csv"
    StressFile = "data/stress_mainlamp.csv"
    Gang2File = "data/gangnum_2.csv"
    Gang3File = "data/gangnum_3.csv"

    # print("File:" + BookingFile)
    booking_name = BookingFile.split("/")[1].split(".")[0]
    # 注文情報の読み込み
    T, L, D, J, U, A, G, J_small, J_medium, J_large, Port, Check_port, Booking, divide_dic\
        = read_booking.Read_booking(BookingFile)

    # 船体情報の読み込み1
    I, B, I_pair, I_next, I_same, I_lamp, I_deck, RT_benefit, delta_s, min_s, max_s, delta_h, max_h, Hold_encode, Hold\
        = read_hold.Read_hold(HoldFile)
    
    # 船体情報の読み込み2
    Ml_Load, Ml_Back, Ml_Afr, Stress, GANG2, GANG3\
        = read_other.Read_other(MainLampFile, BackMainLampFile, AfrMainLampFile, StressFile, Gang2File, Gang3File, Hold_encode)
        
    filling_rate = read_hold.Read_other(AfrFile)
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
    
    # デッドスペースのペナルティ
    penal5_k = 1000
    
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


        
    def assign_to_hold(assignment_list,find_separation=False,shift_next_segment = -1,order_loading_port= -1):
        hold_assignment = []
        for i in range(HOLD_COUNT):
            hold_assignment.append([])
        unloaded_orders = []
        half_way_loaded_rt = [] #最終港以外で，積みを終えたときのRTの配列
        half_way_assignments = [] #1港目を積み終えたときのバランスを計算するための配列
        for loading_port_num in range(len(L)-1):     
            tmp = {}
            tmp2 = []
            for hold_num in range(HOLD_COUNT):
                tmp[hold_num] = []
                tmp2.append(0)
            half_way_assignments.append(tmp)
            half_way_loaded_rt.append(tmp2)
        
        balance_penalty = 0
        
        separation_orders = []

        for segment_num in range(SEGMENT_COUNT):
            segment = segments[segment_num]
            assignment = assignment_list[segment_num]
            
            assignment_RT = []
            assignment_unit = []
            assignment_total_space = []
            
            for loading_port_num in range(len(assignment)):
                tmp_assignment_RT = []
                tmp_assignment_unit = []
                tmp_assignment_total_space = []
                for order in assignment[loading_port_num]:
                    tmp_assignment_RT.append(A[order])
                    tmp_assignment_unit.append(int(U[order]))
                    tmp_assignment_total_space.append(A[order]*int(U[order]))
                assignment_RT.append(tmp_assignment_RT)
                assignment_unit.append(tmp_assignment_unit)
                assignment_total_space.append(tmp_assignment_total_space)
            
            left_spaces = {}
            for hold in segment:
                left_spaces[hold] = B[hold]
            
            
            for loading_port_num in range(len(assignment)):
                orders = assignment[loading_port_num]
                order_cnt = 0 #どの注文まで積んだか
                orders_size = len(orders) #注文の個数
                if loading_port_num == len(assignment)-1: #最後に積む港
                    for hold in segment:
                        # 全部詰め切るか，そのホールドに注文をまるごと詰め込めなくなったらwhileを抜ける
                        while (order_cnt < orders_size and assignment_total_space[loading_port_num][order_cnt]<left_spaces[hold]):
                            left_spaces[hold] -= assignment_total_space[loading_port_num][order_cnt]
                            hold_assignment[hold].append([assignment[loading_port_num][order_cnt],assignment_unit[loading_port_num][order_cnt]])
                            assignment_total_space[loading_port_num][order_cnt] = 0
                            assignment_unit[loading_port_num][order_cnt] = 0
                            order_cnt += 1
                        # まるごとは注文を詰め込めなくても，一部なら可能なら一部を詰め込む
                        if order_cnt < orders_size:
                            if find_separation:
                                if shift_next_segment == segment_num and order_loading_port==loading_port_num:
                                    separation_orders.append(assignment[loading_port_num][order_cnt])
                            possible_unit_cnt = int(left_spaces[hold] // assignment_RT[loading_port_num][order_cnt])
                            if (possible_unit_cnt>0):
                                left_spaces[hold] -= assignment_RT[loading_port_num][order_cnt] * possible_unit_cnt
                                hold_assignment[hold].append([assignment[loading_port_num][order_cnt],possible_unit_cnt])
                                assignment_total_space[loading_port_num][order_cnt] -= assignment_RT[loading_port_num][order_cnt] * possible_unit_cnt
                                assignment_unit[loading_port_num][order_cnt] -= possible_unit_cnt
                else: #最後に積む港ではない場合
                    for hold in segment:
                        ALLOWANCE_SPACE = B[hold] *  (1-filling_rate[hold])
                        ALLOWANCE_STRESS_SPACE = B[hold] * (1-Stress[hold])
                        # 全部詰め切るか，そのホールドに作業効率充填率を満たしつつ注文をまるごと詰め込めなくなったらwhileを抜ける
                        while (order_cnt < orders_size and assignment_total_space[loading_port_num][order_cnt]<(left_spaces[hold])-ALLOWANCE_STRESS_SPACE):
                            left_spaces[hold] -= assignment_total_space[loading_port_num][order_cnt]
                            hold_assignment[hold].append([assignment[loading_port_num][order_cnt],assignment_unit[loading_port_num][order_cnt]])
                            half_way_assignments[loading_port_num][hold].append([assignment[loading_port_num][order_cnt],assignment_unit[loading_port_num][order_cnt]])
                            assignment_total_space[loading_port_num][order_cnt] = 0
                            assignment_unit[loading_port_num][order_cnt] = 0
                            order_cnt += 1
                            
                        # まるごとは注文を詰め込めなくても，作業効率充填率を満たしつつ一部なら可能なら一部を詰め込む
                        if order_cnt < orders_size:
                            if find_separation:
                                if shift_next_segment == segment_num and order_loading_port==loading_port_num:
                                    separation_orders.append(assignment[loading_port_num][order_cnt])
                            possible_unit_cnt = int((left_spaces[hold]-ALLOWANCE_STRESS_SPACE) // assignment_RT[loading_port_num][order_cnt])
                            if (possible_unit_cnt>0):
                                left_spaces[hold] -= assignment_RT[loading_port_num][order_cnt] * possible_unit_cnt
                                hold_assignment[hold].append([assignment[loading_port_num][order_cnt],possible_unit_cnt])
                                half_way_assignments[loading_port_num][hold].append([assignment[loading_port_num][order_cnt],possible_unit_cnt])
                                assignment_total_space[loading_port_num][order_cnt] -= assignment_RT[loading_port_num][order_cnt] * possible_unit_cnt
                                assignment_unit[loading_port_num][order_cnt] -= possible_unit_cnt
                        
                        # 全部詰め切るか，そのホールドに注文をまるごと詰め込めなくなったらwhileを抜ける
                        while (order_cnt < orders_size and assignment_total_space[loading_port_num][order_cnt]<(left_spaces[hold])-ALLOWANCE_SPACE):
                            left_spaces[hold] -= assignment_total_space[loading_port_num][order_cnt]
                            hold_assignment[hold].append([assignment[loading_port_num][order_cnt],assignment_unit[loading_port_num][order_cnt]])
                            half_way_assignments[loading_port_num][hold].append([assignment[loading_port_num][order_cnt],assignment_unit[loading_port_num][order_cnt]])
                            assignment_total_space[loading_port_num][order_cnt] = 0
                            assignment_unit[loading_port_num][order_cnt] = 0
                            order_cnt += 1
                        
                        # まるごとは注文を詰め込めなくても，一部なら可能なら一部を詰め込む
                        if order_cnt < orders_size:
                            if find_separation:
                                if shift_next_segment == segment_num and order_loading_port==loading_port_num:
                                    separation_orders.append(assignment[loading_port_num][order_cnt])
                            possible_unit_cnt = int((left_spaces[hold]-ALLOWANCE_SPACE) // assignment_RT[loading_port_num][order_cnt])
                            if (possible_unit_cnt>0):
                                left_spaces[hold] -= assignment_RT[loading_port_num][order_cnt] * possible_unit_cnt
                                hold_assignment[hold].append([assignment[loading_port_num][order_cnt],possible_unit_cnt])
                                half_way_assignments[loading_port_num][hold].append([assignment[loading_port_num][order_cnt],possible_unit_cnt])
                                assignment_total_space[loading_port_num][order_cnt] -= assignment_RT[loading_port_num][order_cnt] * possible_unit_cnt
                                assignment_unit[loading_port_num][order_cnt] -= possible_unit_cnt
                        half_way_loaded_rt[loading_port_num][hold] = B[hold] - left_spaces[hold]
                for index in range(len(assignment_unit[loading_port_num])): 
                    if assignment_unit[loading_port_num][index]>0:
                        unloaded_orders.append([orders[index],assignment_unit[loading_port_num][index]])
        # バランス制約を計算
        # 最後に積む港以外
        balance1 = 0
        balance2 = 0
        for loading_port_num in range(len(L)-1):     
            half_way_assignment = half_way_assignments[loading_port_num]
            for hold_num,orders in half_way_assignment.items():
                if len(orders)>0:
                    for order in orders:
                        #横方向
                        balance1 +=delta_h[hold_num] * G[order[0]] * order[1]
                        # balance_penalty += max(0,(delta_h[hold_num] * G[order[0]] * order[1]) - max_h)
                        #縦方向
                        balance2 += delta_s[hold_num] * G[order[0]] * order[1]
                        # balance_penalty += max(0,(delta_s[hold_num] * G[order[0]] * order[1]) - max_s)
                        # balance_penalty += max(0,min_s-(delta_s[hold_num] * G[order[0]] * order[1]))
            balance_penalty += max(0,balance1-max_h)
            balance_penalty += max(0,balance2-max_s)
            balance_penalty += max(0,min_s-balance2)
        
        balance1 = 0
        balance2 = 0
                
        # 最後に積む港
        for hold_num in range(len(hold_assignment)):
            tmp_assignment = hold_assignment[hold_num]
            if len(tmp_assignment)>0:
                for order in tmp_assignment:
                    #横方向
                    balance1 +=delta_h[hold_num] * G[order[0]] * order[1]
                    # balance_penalty += max(0,(delta_h[hold_num] * G[order[0]] * order[1]) - max_h)
                    #縦方向
                    balance2 += delta_s[hold_num] * G[order[0]] * order[1]
                    # balance_penalty += max(0,(delta_s[hold_num] * G[order[0]] * order[1]) - max_s)
                    # balance_penalty += max(0,min_s-(delta_s[hold_num] * G[order[0]] * order[1]))        
        balance_penalty += max(0,balance1-max_h)
        balance_penalty += max(0,balance2-max_s)
        balance_penalty += max(0,min_s-balance2)
        """
        返り値は，2次元の配列
        hold_assignment[i]で，ホールドiに割り当てられる注文の情報の配列を取得できる
        hold_assignment[i][j]で，ホールドiにj番目に割り当てられる注文を取得できる
        配列の0番目が，注文の番号 
        配列の1番目が，割り当てる台数
        """
        
        separation_orders = set(separation_orders)
        separation_index = []
        for order in separation_orders:
            separation_index.append(assignment_list[shift_next_segment][order_loading_port].index(order))
        if find_separation == False:
            return hold_assignment,unloaded_orders,balance_penalty,half_way_loaded_rt
        else:
            return hold_assignment,unloaded_orders,balance_penalty,half_way_loaded_rt,separation_index
    
    def evaluate(assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt):
        # 全注文内の自動車の台数を全て割り当てる
        n_it = []
        total_left_RT = 0
        for hold_num in range(HOLD_COUNT):
            assignment = assignment_hold[hold_num]
            left_RT = B[hold_num]
            for order in assignment:
                left_RT -= A[order[0]]*order[1]
            n_it.append(left_RT)  
            total_left_RT += left_RT
        unloaded_units = 0
        for order in unloaded_orders:
            unloaded_units += order[1]
        # ここまで
        
        constraint1 = 0 #移動経路制約
        
        objective3 = 0 #作業効率充填率
        # 縦横方向のバランス制約
        balance_constraint1 = [0 for i in range(len(D))]
        balance_constraint2 = [0 for i in range(len(D))]
        
        # 経路確保とバランス制約の計算に使う配列を作成
        for hold_num in range(len(assignment_hold)):         
            assignment_in_hold = assignment_hold[hold_num]
            destination_assignments = [[] for i in range(len(D))]
            for assign in assignment_in_hold:
                destination_port = int(Booking.at[assign[0],"DPORT"])
                if (destination_port-len(L)!=0):
                    for i in range(1,destination_port-len(L)+1):
                        destination_assignments[i].append(assign)  
                                   
            #降ろし地での経路確保  降ろし地での作業効率充填率
            hold_space = B[hold_num]
            for i in range(1,len(destination_assignments)):
                total_loaded_space = 0
                for assign in destination_assignments[i]:
                    total_loaded_space += A[assign[0]]*assign[1]
                if total_loaded_space > hold_space*filling_rate[hold_num]:
                    constraint1 += (total_loaded_space-(hold_space*filling_rate[hold_num]))
                # if total_loaded_space > hold_space*Stress[hold_num]:
                #     objective3 += (total_loaded_space-(hold_space*Stress[hold_num]))
            
            #ここまで
            
            #降ろし地でのバランス制約
            for destination_load_num in range(1,len(destination_assignments)):
                for assign in destination_assignments[destination_load_num]:
                    balance_constraint1[destination_load_num] += delta_h[hold_num] * G[assign[0]] * assign[1]
                    balance_constraint2[destination_load_num] += delta_s[hold_num] * G[assign[0]] * assign[1]
            #ここまで
            
            # 降ろし地での作業効率充填率
            order_array = []
            accpeted_rate = Stress[hold_num]
            for port_num in range(len(T)):    
                order_array.append([])
            for item in assignment_in_hold:
                lport = Booking.at[item[0],"LPORT"]
                for i in range(int(lport),int(L[-1])+1):
                    order_array[i].append(item)
                dport = Booking.at[item[0],"DPORT"]
                for i in range(int(D[0]),int(dport)):
                    order_array[i].append(item)

            for port in check_port:
                order = order_array[port]
                total_RT = 0
                for item in order:
                    single_rt = Booking.at[item[0],"RT"]
                    total_RT += item[1]*single_rt
                if total_RT > B[hold_num]*accpeted_rate:
                    objective3 += total_RT - (B[hold_num]*accpeted_rate)
            
        #全ての降ろし地での全てのバランス制約を計算したので，制約違反していたらペナルティ 
        for destination_load_num in range(1,len(destination_assignments)):          
            balance_penalty += max(0,balance_constraint1[destination_load_num]-max_h)
            balance_penalty += max(0,balance_constraint2[destination_load_num]-max_s)
            balance_penalty += max(0,min_s-balance_constraint2[destination_load_num])
        #ここまで
        
        
        objective1 = 0
        # 目的関数1 ひとつのホールドで，異なる降ろし地の注文を少なくする
        for each_assignment in assignment_hold:
            different_destination_area_orders = []
            for order in each_assignment:
                different_destination_area_orders.append(Booking.at[order[0],"DPORT"])
            unique_destination_ports = set(different_destination_area_orders)
            if (len(unique_destination_ports)>1):
                objective1 += penal1_z * (len(unique_destination_ports)-1)
        # ここまで
        
        
        objective2 = 0
        # 目的関数2 注文の積み降ろし地を揃える
        for p in I_pair:
            hold1 = p[0]
            hold2 = p[1]
            orders1 = assignment_hold[hold1]
            orders2 = assignment_hold[hold2]
            lport1 = []
            lport2 = []
            dport1 = []
            dport2 = []
            for order in orders1:
                dport1.append(Booking.at[order[0],"DPORT"])
                lport1.append(Booking.at[order[0],"LPORT"])
            for order in orders2:
                dport2.append(Booking.at[order[0],"DPORT"])
                lport2.append(Booking.at[order[0],"LPORT"])
            dport1 = set(dport1)
            dport2 = set(dport2)
            dport = set()
            if len(dport.union(dport1,dport2)) > 1:
                objective2 += penal2_dis * len(dport.union(dport1,dport2))-1
            lport1 = set(lport1)
            lport2 = set(lport2)
            lport = set()
            if len(lport.union(lport1,lport2)) > 1:
                objective2 += penal2_load * len(lport.union(lport1,lport2))-1
        #ここまで
        
        # 目的関数4 デッドスペースを作らない
        objective4 = 0
        check_port_dead_space = L[:-1]
        for port in check_port_dead_space:
            total_RT = half_way_loaded_rt[port]
            for hold_with_lamp in I_lamp:
                if total_RT[hold_with_lamp] > B[hold_with_lamp]*filling_rate[hold_with_lamp]:
                    for hold in deeper:
                        if B[hold]-  total_RT[hold] >= 1:
                            objective4 += penal5_k
        # ここまで
        
        # 目的関数5 残容量を入り口に寄せる
        objective5 = 0
        for i in range(len(n_it)):
            objective5 += n_it[i] * RT_benefit[i]
        # ここまで    
        return unloaded_units+balance_penalty+constraint1,objective1+objective2+objective3+objective4-objective5
    
    
    def is_feasible(assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt):
        # 全注文内の自動車の台数を全て割り当てる
        n_it = []
        total_left_RT = 0
        for hold_num in range(HOLD_COUNT):
            assignment = assignment_hold[hold_num]
            left_RT = B[hold_num]
            for order in assignment:
                left_RT -= A[order[0]]*order[1]
            n_it.append(left_RT)  
            total_left_RT += left_RT
        unloaded_units = 0
        for order in unloaded_orders:
            unloaded_units += order[1]
        # ここまで
        
        constraint1 = 0 #移動経路制約
        
        objective3 = 0 #作業効率充填率
        # 縦横方向のバランス制約
        balance_constraint1 = [0 for i in range(len(D))]
        balance_constraint2 = [0 for i in range(len(D))]
        
        # 経路確保とバランス制約の計算に使う配列を作成
        for hold_num in range(len(assignment_hold)):         
            assignment_in_hold = assignment_hold[hold_num]
            destination_assignments = [[] for i in range(len(D))]
            for assign in assignment_in_hold:
                destination_port = int(Booking.at[assign[0],"DPORT"])
                if (destination_port-len(L)!=0):
                    for i in range(1,destination_port-len(L)+1):
                        destination_assignments[i].append(assign)  
                                   
            #降ろし地での経路確保  降ろし地での作業効率充填率
            hold_space = B[hold_num]
            for i in range(1,len(destination_assignments)):
                total_loaded_space = 0
                for assign in destination_assignments[i]:
                    total_loaded_space += A[assign[0]]*assign[1]
                if total_loaded_space > hold_space*filling_rate[hold_num]:
                    constraint1 += (total_loaded_space-(hold_space*filling_rate[hold_num]))
                # if total_loaded_space > hold_space*Stress[hold_num]:
                #     objective3 += (total_loaded_space-(hold_space*Stress[hold_num]))
            
            #ここまで
            
            #降ろし地でのバランス制約
            for destination_load_num in range(1,len(destination_assignments)):
                for assign in destination_assignments[destination_load_num]:
                    balance_constraint1[destination_load_num] += delta_h[hold_num] * G[assign[0]] * assign[1]
                    balance_constraint2[destination_load_num] += delta_s[hold_num] * G[assign[0]] * assign[1]
            #ここまで
            
            # 降ろし地での作業効率充填率
            order_array = []
            accpeted_rate = Stress[hold_num]
            for port_num in range(len(T)):    
                order_array.append([])
            for item in assignment_in_hold:
                lport = Booking.at[item[0],"LPORT"]
                for i in range(int(lport),int(L[-1])+1):
                    order_array[i].append(item)
                dport = Booking.at[item[0],"DPORT"]
                for i in range(int(D[0]),int(dport)):
                    order_array[i].append(item)

            for port in check_port:
                order = order_array[port]
                total_RT = 0
                for item in order:
                    single_rt = Booking.at[item[0],"RT"]
                    # print(item[0])
                    # print(Booking.at[item[0]-1,"RT"])

                    total_RT += item[1]*single_rt
                    # print(single_rt,item[1])
                if total_RT > B[hold_num]*accpeted_rate:
                    objective3 += total_RT - (B[hold_num]*accpeted_rate)
            
        #全ての降ろし地での全てのバランス制約を計算したので，制約違反していたらペナルティ 
        for destination_load_num in range(1,len(destination_assignments)):          
            balance_penalty += max(0,balance_constraint1[destination_load_num]-max_h)
            balance_penalty += max(0,balance_constraint2[destination_load_num]-max_s)
            balance_penalty += max(0,min_s-balance_constraint2[destination_load_num])
        #ここまで
        
        
        objective1 = 0
        # 目的関数1 ひとつのホールドで，異なる降ろし地の注文を少なくする
        for each_assignment in assignment_hold:
            different_destination_area_orders = []
            for order in each_assignment:
                different_destination_area_orders.append(Booking.at[order[0],"DPORT"])
            unique_destination_ports = set(different_destination_area_orders)
            if (len(unique_destination_ports)>1):
                objective1 += penal1_z * (len(unique_destination_ports)-1)
        # ここまで
        
        
        objective2 = 0
        # 目的関数2 注文の積み降ろし地を揃える
        for p in I_pair:
            hold1 = p[0]
            hold2 = p[1]
            orders1 = assignment_hold[hold1]
            orders2 = assignment_hold[hold2]
            lport1 = []
            lport2 = []
            dport1 = []
            dport2 = []
            for order in orders1:
                dport1.append(Booking.at[order[0],"DPORT"])
                lport1.append(Booking.at[order[0],"LPORT"])
            for order in orders2:
                dport2.append(Booking.at[order[0],"DPORT"])
                lport2.append(Booking.at[order[0],"LPORT"])
            dport1 = set(dport1)
            dport2 = set(dport2)
            dport = set()
            if len(dport.union(dport1,dport2)) > 1:
                objective2 += penal2_dis * len(dport.union(dport1,dport2))-1
            lport1 = set(lport1)
            lport2 = set(lport2)
            lport = set()
            if len(lport.union(lport1,lport2)) > 1:
                objective2 += penal2_load * len(lport.union(lport1,lport2))-1
        #ここまで
        
        # 目的関数4 デッドスペースを作らない
        objective4 = 0
        check_port_dead_space = L[:-1]
        for port in check_port_dead_space:
            total_RT = half_way_loaded_rt[port]
            for hold_with_lamp in I_lamp:
                if total_RT[hold_with_lamp] > B[hold_with_lamp]*filling_rate[hold_with_lamp]:
                    for hold in deeper:
                        if B[hold]-  total_RT[hold] >= 1:
                            objective4 += penal5_k
        # ここまで
        
        # 目的関数5 残容量を入り口に寄せる
        objective5 = 0
        for i in range(len(n_it)):
            objective5 += n_it[i] * RT_benefit[i]
            
        print(unloaded_units,balance_penalty,constraint1)
        print(objective1,objective2,objective3,objective4,objective5)    
        # ここまで    
        return penalty_coefficient*(unloaded_units+balance_penalty+constraint1)+objective1+objective2+objective3+objective4-objective5
    

    dt1 = datetime.datetime.now()    
    random.seed(1)

    SEGMENT_COUNT = 18
    HOLD_COUNT = 43
    ORDER_COUNT = len(J)
    UNIQUE_ORDER_COUNT = len(set(Booking["Order_num"]))
    check_port = L[:-1] + D[:-1]
    # 分割したホールドで，奥から詰める順番で配列を作成
    segments = np.array([[1, 0],
                         [2, 3],
                         [4],
                         [5, 6, 7],
                         [8, 9, 10],
                         [11, 12, 13, 14],
                         [18, 17, 16, 15],
                         [22, 21, 20, 19],
                         [23, 24],
                         [26, 25],
                         [27, 28],
                         [30, 29],
                         [31, 32],
                         [34, 33],
                         [35, 36],
                         [38, 37],
                         [39, 40],
                         [42, 41]]
                        )
    def segment_index(hold_num):
        for index in range(len(segments)):
            if hold_num in segments[index]:
                return index

    each_segments_size = []
    for i in range(SEGMENT_COUNT):
        total_size = 0
        for j in range(len(segments[i])):
            total_size += B[segments[i][j]]
        each_segments_size.append(total_size)

    '''
    解の持ち方 3次元配列で持つ
    assignment[i]で，セグメントiに割り振られた注文を見れる
    assignment[i][j]で，セグメントiに割り振られた注文の，なかで，j個目の積み地のものを見れる
    assignment[i][j][k]で，j個目の港でk個目に積み込む注文を見れる
    '''
    
    
    # 解の初期化
    assignment = []
    # とりあえず空で初期化
    for i in range(SEGMENT_COUNT):
        tmp = []
        for j in range(len(L)): #積み地の数だけ空配列を追加
            tmp.append([])
        assignment.append(tmp)

    
    # モデル2の解を他のファイルから呼び出し
    model2_assignment = model2_script.model2(BookingFile)
    # print(model2_assignment)
    
    # モデル2の解をもとに初期解を生成する手順
    
    # LPORTとDPORTの2つから、注文を全て見れるデータ構造を作る done
    # ↑で作ったデータ構造で、注文とRT見れるようにする done
    # RTが大きい順に、注文番号を並び替える done
    # モデル2の初期解から、ホールドに割り当てるRTを集計 done
    # ホールドに割り当てる集合の配列を作成 done
    # ホールドに割り当てれる限り割り当てる done
    # セグメントで、割り当てたRTの合計を計算する done
    # 空きRTが多いところから、未割り当ての注文を割り当てる done
    
    # 未割り当ての注文が残ったら、ランダムに割り当てる
    
    # order_list_by_port[LPORT][DPORT]で、積み地と揚げ地に対応する注文を全て見れる
    order_list_by_port = []
    for lport in L:
        order_list_by_port.append([])
    
    for i in range(len(order_list_by_port)):
        for dport in T:
            order_list_by_port[i].append([])
    
    for index in range(len(Booking)):
        lport_num = int(Booking.at[index,"LPORT"])
        dport_num = int(Booking.at[index,"DPORT"])
        single_rt = Booking.at[index,"RT"]
        unit = int(Booking.at[index,"Units"])
        total_rt = single_rt*unit
        order_list_by_port[lport_num][dport_num].append([index,total_rt])
    
    for i in range(len(order_list_by_port)):
        for j in range(len(order_list_by_port[i])):
            order_list_by_port[i][j] = sorted(order_list_by_port[i][j], reverse=True, key=lambda x: x[1])
    
    # for i in range(len(order_list_by_port)):
    #     for j in range(len(order_list_by_port[i])):
    #         print(order_list_by_port[i][j])
            
    # model2_rt_by_hold[ホールド番号][LPort][DPort]で、ホールドに割り当てるRTを見れる
    model2_rt_by_hold = []
    for hold_num in range(HOLD_COUNT):
        model2_rt_by_hold.append([])
        for lport_num in range(len(L)):
            model2_rt_by_hold[hold_num].append([])
            for dport in T:
                model2_rt_by_hold[hold_num][lport_num].append(0)
    

    # model2_rt_by_holdの初期化終わり
    tmp = 0
    for index in range(len(model2_assignment)):
        hold_id = model2_assignment.at[index,"Hold_ID"]
        lport_num = model2_assignment.at[index,"LPORT"]
        dport_num = model2_assignment.at[index,"DPORT"]
        load_rt = model2_assignment.at[index,"Load_RT"]
        model2_rt_by_hold[hold_id][lport_num][dport_num] += load_rt
        tmp += load_rt
    # model2_rt_by_holdの値の挿入終わり
    
    # ホールドに割り当て 
    # ホールドに割り当てられる集合の配列を作成、初期化
    model2_hold_assignment = []
    for hold_num in range(HOLD_COUNT):
        model2_hold_assignment.append([])
        for lport_num in range(len(L)):
            model2_hold_assignment[hold_num].append([])
    
    # 実際に割り当て
    for hold_num in range(HOLD_COUNT):
        for lport_num in range(len(model2_rt_by_hold[hold_num])):
            for dport_num in range(len(model2_rt_by_hold[hold_num][lport_num])):
                if model2_rt_by_hold[hold_num][lport_num][dport_num]>0 and len(order_list_by_port[lport_num][dport_num])>0:
                    # print(order_list_by_port[lport_num][dport_num])
                    rest_rt = model2_rt_by_hold[hold_num][lport_num][dport_num]
                    #挿入したインデックスのリスト(deleted_index_list)を作成
                    #i番目の注文のRTが、残りより多い時は挿入
                        # model2_hold_assignment[hold_num][lport_num]に追加
                        # 挿入したら、deleted_index_listにインデックスを追加
                        # model2_rt_by_holdの残りRTを減らす
                    # 大きすぎて割り当てできないものは、スキップ
                    # 最後まで見たら終わり
                        #追加した注文を、order_list_by_portから削除
                    
                    deleted_index_list = []
                    for check_index in range(0,len(order_list_by_port[lport_num][dport_num])):
                        if order_list_by_port[lport_num][dport_num][check_index][1]<rest_rt:
                            model2_hold_assignment[hold_num][lport_num].append(order_list_by_port[lport_num][dport_num][check_index][0])
                            deleted_index_list.append(check_index)
                            rest_rt -= order_list_by_port[lport_num][dport_num][check_index][1]
                    model2_rt_by_hold[hold_num][lport_num][dport_num] = rest_rt
                    for deleted_index in reversed(deleted_index_list):
                        order_list_by_port[lport_num][dport_num].pop(deleted_index)
    # for item in model2_rt_by_hold:
    #     print(item)
        
    # ホールドからセグメントへの変換
    # rest_rt_by_segment[セグメント番号][LPort][DPort]で、まだ詰めるRTがわかる
    rest_rt_by_segment = []
    for i in range(len(segments)): #セグメントの数
        rest_rt_by_segment.append([])
        for lport_num in range(len(L)): #積み地
            rest_rt_by_segment[i].append([])
            for dport in T: #揚げ地
                rest_rt_by_segment[i][lport_num].append(0) #0で初期化
    
        
    for hold_num in range(HOLD_COUNT):
        for lport_num in range(len(model2_rt_by_hold[hold_num])):
            for dport_num in range(len(model2_rt_by_hold[hold_num][lport_num])):
                   if model2_rt_by_hold[hold_num][lport_num][dport_num]>0:
                        segment_id = segment_index(hold_num) 
                        rest_rt_by_segment[segment_id][lport_num][dport_num] += model2_rt_by_hold[hold_num][lport_num][dport_num]
    
        
    # ホールドへの割り当てから、セグメントでの割り当てに変換
    for hold_num in range(HOLD_COUNT):
        segment_id = segment_index(hold_num) 
        for lport_num in range(len(L)):
            for order in model2_hold_assignment[hold_num][lport_num]:
                assignment[segment_id][lport_num].append(order)
    # for item in assignment:
    #     print(item)

    
    # todo
    # model2_rt_by_holdを、lport,dportで縦刺しに見て、model2_rt_by_hold[lport_num][dport_num]の空き容量が多い順に並び替える
    # order_list_by_port[lport_num][dport_num]で、未割り当ての注文が見れるから、RTの大きい未割り当ての注文から割り当てる
    # それでも割り当てできないものは、空き容量が多いセグメントにランダムに割り当てる
    
    for lport_num in range(len(order_list_by_port)):
        for dport_num in range(len(order_list_by_port[lport_num])):
            if len(order_list_by_port[lport_num][dport_num])>0: #未割り当ての注文がある場合のみ考える
                deleted_index_list = []
                for unassigned_order_cnt in range(len(order_list_by_port[lport_num][dport_num])): 
                    unassigned_order = order_list_by_port[lport_num][dport_num][unassigned_order_cnt]
                    order_num = unassigned_order[0]
                    rt = unassigned_order[1]
                    max_rt_segment_num = 0
                    max_rt = rest_rt_by_segment[0][lport_num][dport_num] 
                    for segment_num in range(1,len(rest_rt_by_segment)):
                        rest_rt = rest_rt_by_segment[segment_num][lport_num][dport_num]
                        if rest_rt > max_rt:
                            max_rt = rest_rt
                            max_rt_segment_num = segment_num
                    
                    # ここまでで、空き容量が一番多いセグメントを選んだ
                    # 割り当てることができたら割り当てる   
                    if rt<max_rt:
                        assignment[max_rt_segment_num][lport_num].append(order_num)
                        deleted_index_list.append(unassigned_order_cnt)
                        rest_rt_by_segment[max_rt_segment_num][lport_num][dport_num] -= rt
                
                for deleted_index in reversed(deleted_index_list):
                    order_list_by_port[lport_num][dport_num].pop(deleted_index)
                 
    for lport_num in range(len(order_list_by_port)):
        for dport_num in range(len(order_list_by_port[lport_num])):
            if len(order_list_by_port[lport_num][dport_num])>0: #未割り当ての注文がある場合のみ考える 
                order_rt_pair = []      
                for segment_num in range(len(rest_rt_by_segment)):
                    order_rt_pair.append([segment_num,rest_rt_by_segment[segment_num][lport_num][dport_num]])
                order_rt_pair = sorted(order_rt_pair, reverse=True, key=lambda x: x[1])
    
                order_rt_pair_index = 0
                for unassigned_order_cnt in range(len(order_list_by_port[lport_num][dport_num])):
                    unassigned_order = order_list_by_port[lport_num][dport_num][unassigned_order_cnt]
                    assignment[order_rt_pair[order_rt_pair_index%SEGMENT_COUNT][0]][lport_num].append(unassigned_order[0])
                    order_rt_pair_index += 1
    
    
    

    # # LPORTとDPORTの2つから、注文を全て見れるデータ構造を作る done
    # # 注文をシャッフルする
    # # セグメントごとに、合計RTを集計する done
    # # 合計RTを満たすまで、ランダムにした注文を初期解に詰め込んでいく assignment[セグメント番号][積み地の番号]に追加
    
    
    # # order_list_by_port[LPORT][DPORT]で、積み地と揚げ地に対応する注文を全て見れる
    # order_list_by_port = []
    # for lport in L:
    #     order_list_by_port.append([])
    
    # for i in range(len(order_list_by_port)):
    #     for dport in T:
    #         order_list_by_port[i].append([])
    # for index in range(len(Booking)):
    #     lport = int(Booking.at[index,"LPORT"])
    #     dport = int(Booking.at[index,"DPORT"])
    #     order_list_by_port[lport][dport].append(index)
        
        
    # for i in range(len(order_list_by_port)):
    #     for j in range(len(order_list_by_port[i])):
    #         order_list_by_port[i][j].sort(reverse=True)
    
    # for i in range(len(order_list_by_port)):
    #     for j in range(len(order_list_by_port[i])):
    #         print(order_list_by_port[i][j])

    # # セグメントと積み地と揚げ地に対応する多次元配列を作る arr[セグメント][lport][dport]に、対応するRTとかかな done
    
    # initial_rt_by_segment = []
    # for segment in segments:
    #     initial_rt_by_segment.append([])
    
    # for i in range(len(initial_rt_by_segment)): #セグメントの数
    #     for lport_num in range(len(L)): #積み地
    #         initial_rt_by_segment[i].append([])
    #         for dport in T: #揚げ地
    #             initial_rt_by_segment[i][lport_num].append(0) #0で初期化
    
    # for index in range(len(model2_assignment)):
    #     hold_id = model2_assignment.at[index,"Hold_ID"]
    #     segment_id = segment_index(hold_id)  
    #     lport = model2_assignment.at[index,"LPORT"]
    #     dport = model2_assignment.at[index,"DPORT"]
    #     load_rt = model2_assignment.at[index,"Load_RT"]
    #     initial_rt_by_segment[segment_id][lport][dport] += load_rt
    
    # for item in initial_rt_by_segment:
    #     print(item)
    # print(Booking)
    
    # print("----")
    # for segment_num in range(len(segments)):
    #     print(segment_num)
    #     print(initial_rt_by_segment[segment_num])
    #     for lport in range(len(initial_rt_by_segment[segment_num])):
    #         for dport in range(len(initial_rt_by_segment[segment_num][lport])):
    #             initial_rt_by_segment[segment_num][lport][dport]
    #             while initial_rt_by_segment[segment_num][lport][dport]>0: #積まれるべき合計RTがまだある場合
    #                 if len(order_list_by_port[lport][dport]) >0: #まだ積まれていない注文が残っている場合
    #                     order = order_list_by_port[lport][dport].pop()
    #                     # print(order)
    #                     # print(Booking.at[order,"Units"],Booking.at[order,"RT"])
    #                     tmp_rt = Booking.at[order,"Units"]*Booking.at[order,"RT"]
    #                     # print(tmp_rt)
    #                     initial_rt_by_segment[segment_num][lport][dport] -= tmp_rt
    #                     assignment[segment_num][lport].append(order)
    #                 else:
    #                     break
    #     print(initial_rt_by_segment[segment_num])
        
        

    # for item in order_list_by_port:
    #     print(item)

    # for i in range(len(L)):
    #     randomed_J = random.sample(J_t_load[i], len(J_t_load[i]))
    #     for j in range(len(randomed_J)):
    #         assignment[j%SEGMENT_COUNT][i].append(randomed_J[j])

    penalty_coefficient = 100
    #初期解を，ホールドに割当
    assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt = assign_to_hold(assignment)
    #初期解のペナルティ    
    penalty,objective = evaluate(assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt)
    evaluated_value = penalty_coefficient*penalty+objective
    print(evaluated_value)
    shift_neighbor_list = operation.create_shift_neighbor(ORDER_COUNT,SEGMENT_COUNT)
    shift_count = 0
    # swap_neighbor_list = operation.create_swap_neighbor(J_t_load,Booking)
    swap_neighbor_list = operation.create_optimized_swap_neighbor(J_t_load,Booking)
    swap_count = 0
    total_improve = 1
    
    
    

    # print(assignment[4])
    # print('----')
    # assignment= operation.shift(assignment,170,0,4,1,5)
    # print(assignment[4])
    # print('----')
    # assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt,separation_index = assign_to_hold(assignment,True,4,1)
    # for i in range(8,11):
    #     print(assignment_hold[i])
    # print(separation_index)
    
    """
    ここからテスト
    # shift_order = 17
    # shift_seg = 1
    # # # 注文29をセグメント1に移動する
    # loading_port = operation.find_loading_port(shift_order,J_t_load)
    # current_segment,current_index = operation.find_current_segment_and_index(assignment,shift_order,loading_port)
    # inserted_index = random.randint(0,len(assignment[shift_seg][loading_port]))
    # assignment= operation.shift(assignment,shift_order,current_segment,shift_seg,loading_port,inserted_index)
    # for item in assignment:
    #     print(item)
    # print("----")
    # assignment= operation.shift(assignment,shift_order,shift_seg,current_segment,loading_port,current_index)
    # for item in assignment:
    #     print(item)
    # assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt = assign_to_hold(assignment)
    # penalty,objective = evaluate(assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt)
    # evaluated_value = penalty_coefficient*penalty+objective
    # print(evaluated_value)
    
    # swap_order1 = 29
    # swap_order2 = 31
    # loading_port = operation.find_loading_port(swap_order1,J_t_load)
    # order1_seg,order1_index = operation.find_current_segment_and_index(assignment,swap_order1,loading_port)
    # order2_seg,order2_index = operation.find_current_segment_and_index(assignment,swap_order2,loading_port)
    # assignment = operation.swap(assignment,swap_order1,order1_seg,order1_index,swap_order2,order2_seg,order2_index,loading_port)
    # #戻す
    # assignment = operation.swap(assignment,swap_order2,order1_seg,order1_index,swap_order1,order2_seg,order2_index,loading_port)
    ここまで
    """
    for item in assignment:
        print(item)
    while total_improve != 0:
        shift_count = 0
        while(shift_count < len(shift_neighbor_list)):
            shift_order = shift_neighbor_list[shift_count][0]
            shift_seg = shift_neighbor_list[shift_count][1]
            loading_port = operation.find_loading_port(shift_order,J_t_load)
            # """
            #ランダムな場所に追加
            # print(shift_order,loading_port)
            # print(operation.find_current_segment_and_index(assignment,shift_order,loading_port))
            # print("---------")
            current_segment,current_index = operation.find_current_segment_and_index(assignment,shift_order,loading_port)
            inserted_index = random.randint(0,len(assignment[shift_seg][loading_port]))
            assignment= operation.shift(assignment,shift_order,current_segment,shift_seg,loading_port,inserted_index)
            assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt = assign_to_hold(assignment)
            tmp_penalty,tmp_objective = evaluate(assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt)
            tmp_evaluated_value = penalty_coefficient*tmp_penalty+tmp_objective
            if  tmp_evaluated_value < evaluated_value:
                print("改善 shift "+str(tmp_evaluated_value))
                evaluated_value= tmp_evaluated_value
                # 探索リストを最初からやり直し
                shift_count = 0 
                random.shuffle(shift_neighbor_list)
            else:
                shift_count += 1
                # 探索の解を戻す
                assignment= operation.shift(assignment,shift_order,shift_seg,current_segment,loading_port,current_index)
            # """
            """
            #全ての場所に挿入して比較
            best_index = -1
            tmp_best_evaluated_value = evaluated_value
            for idx in range(len(assignment[shift_seg][loading_port])+1):
                current_segment,current_index = operation.find_current_segment_and_index(assignment,shift_order,loading_port)
                inserted_index = idx
                assignment= operation.shift(assignment,shift_order,current_segment,shift_seg,loading_port,inserted_index)
                assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt = assign_to_hold(assignment)
                tmp_penalty,tmp_objective = evaluate(assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt)
                tmp_evaluated_value = penalty_coefficient*tmp_penalty+tmp_objective
                if  tmp_evaluated_value < tmp_best_evaluated_value:
                    tmp_best_evaluated_value= tmp_evaluated_value
                    best_index = idx
                assignment= operation.shift(assignment,shift_order,shift_seg,current_segment,loading_port,current_index)
            if best_index!= -1:
                shift_count = 0 
                random.shuffle(shift_neighbor_list)
                assignment= operation.shift(assignment,shift_order,current_segment,shift_seg,loading_port,best_index)
                assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt = assign_to_hold(assignment)
                tmp_penalty,tmp_objective = evaluate(assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt)
                evaluated_value = penalty_coefficient*tmp_penalty+tmp_objective
                print("改善 shift "+str(evaluated_value))
            else:
                shift_count += 1
            """
            """
            #効果のありそうなとろこに挿入して比較
            best_index = -1
            tmp_best_evaluated_value = evaluated_value
            
            #まず一番後ろに挿入
            current_segment,current_index = operation.find_current_segment_and_index(assignment,shift_order,loading_port)
            inserted_index = len(assignment[shift_seg][loading_port])
            assignment= operation.shift(assignment,shift_order,current_segment,shift_seg,loading_port,inserted_index)
            assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt,separation_index = assign_to_hold(assignment,True,shift_seg,loading_port)
            tmp_penalty,tmp_objective = evaluate(assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt)
            tmp_evaluated_value = penalty_coefficient*tmp_penalty+tmp_objective
            if  tmp_evaluated_value < tmp_best_evaluated_value:
                tmp_best_evaluated_value= tmp_evaluated_value
                best_index = inserted_index
            assignment= operation.shift(assignment,shift_order,shift_seg,current_segment,loading_port,current_index)
            insert_index_list= []
            # print(separation_index)
            if len(separation_index)==0:
                insert_index_list.append(0)    
            for item in separation_index:
                if item==0:
                    insert_index_list.append(0)
                elif item != len(assignment[shift_seg][loading_port]):
                    if item ==len(assignment[shift_seg][loading_port])-1:
                        insert_index_list.append(item)
                        insert_index_list.append(item-1)
                    else:
                        insert_index_list.append(item)
                        insert_index_list.append(item+1)
                        insert_index_list.append(item-1)
            insert_index_list =  list(set(insert_index_list))
            # print('change')
            # print(insert_index_list)
            # 効果のありそうなところに挿入
            for idx in insert_index_list:
                current_segment,current_index = operation.find_current_segment_and_index(assignment,shift_order,loading_port)
                inserted_index = idx
                assignment= operation.shift(assignment,shift_order,current_segment,shift_seg,loading_port,inserted_index)
                assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt = assign_to_hold(assignment)
                tmp_penalty,tmp_objective = evaluate(assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt)
                tmp_evaluated_value = penalty_coefficient*tmp_penalty+tmp_objective
                if  tmp_evaluated_value < tmp_best_evaluated_value:
                    tmp_best_evaluated_value= tmp_evaluated_value
                    best_index = idx
                assignment= operation.shift(assignment,shift_order,shift_seg,current_segment,loading_port,current_index)
            
            if best_index!= -1:
                shift_count = 0 
                random.shuffle(shift_neighbor_list)
                assignment= operation.shift(assignment,shift_order,current_segment,shift_seg,loading_port,best_index)
                assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt = assign_to_hold(assignment)
                tmp_penalty,tmp_objective = evaluate(assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt)
                evaluated_value = penalty_coefficient*tmp_penalty+tmp_objective
                print("改善 shift "+str(evaluated_value))
            else:
                shift_count += 1
            """
            
        total_improve = 0
        swap_count = 0
        
        
        while(swap_count < len(swap_neighbor_list)):
            swap_order1 = swap_neighbor_list[swap_count][0]
            swap_order2 = swap_neighbor_list[swap_count][1]
            loading_port = operation.find_loading_port(swap_order1,J_t_load)
            order1_seg,order1_index = operation.find_current_segment_and_index(assignment,swap_order1,loading_port)
            order2_seg,order2_index = operation.find_current_segment_and_index(assignment,swap_order2,loading_port)
            assignment = operation.swap(assignment,swap_order1,order1_seg,order1_index,swap_order2,order2_seg,order2_index,loading_port)
            assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt = assign_to_hold(assignment)
            tmp_penalty,tmp_objective = evaluate(assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt)
            tmp_evaluated_value = penalty_coefficient*tmp_penalty+tmp_objective 
            if  tmp_evaluated_value < evaluated_value:
                print("改善 swap  "+str(tmp_evaluated_value))
                evaluated_value= tmp_evaluated_value
                # 探索リストを最初からやり直し
                random.shuffle(swap_neighbor_list)
                total_improve += 1
                swap_count = 0
                break
            else:
                assignment = operation.swap(assignment,swap_order2,order1_seg,order1_index,swap_order1,order2_seg,order2_index,loading_port)
                swap_count += 1
        
        
    assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt = assign_to_hold(assignment)
    penalty,objective = evaluate(assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt)
    print(penalty*penalty_coefficient+objective)
    print("----")
    output = is_feasible(assignment_hold,unloaded_orders,balance_penalty,half_way_loaded_rt)
    print(output)
    

    dt2 = datetime.datetime.now()
    print("計算時間: "+str((dt2-dt1).total_seconds())+"秒")
    
    result = [["Hold_ID","Order_ID","Load_Units","Units","RT","LPORT","DPORT"]]
    for index in range(len(assignment_hold)):
        assignment_dict = {}
        other_info_dict = {}
        each_assignment = assignment_hold[index]
        for item in each_assignment:
            original_order_num = int(Booking.at[item[0],"Order_num"])
            other_info_dict[original_order_num] = []
            original_units = int(Booking.at[item[0],"Units"])
            original_RT = Booking.at[item[0],"RT"]
            lport = int(Booking.at[item[0],"LPORT"])
            dport = int(Booking.at[item[0],"DPORT"])
            other_info_dict[original_order_num].append(original_units)
            other_info_dict[original_order_num].append(original_RT)
            other_info_dict[original_order_num].append(lport)
            other_info_dict[original_order_num].append(dport)
            if original_order_num in assignment_dict:
                assignment_dict[original_order_num] += item[1]
            else:
                assignment_dict[original_order_num] = item[1]
        hold_id = int(Hold_encode[Hold_encode["Index"]==index]["Hold"])
        for order_id, load_unit in assignment_dict.items():
            info = [hold_id,order_id,load_unit]
            info.extend(other_info_dict[order_id])
            result.append(info)
    out_df = pd.DataFrame(result)
    booking_name = BookingFile.split("/")[1].split(".")[0]
    # assignment_file_name = "out/"+booking_name+"_assignment.xlsx"
    # out_df.to_excel(assignment_file_name,index=False,header=False)            
     
if __name__ == "__main__":
    main()
