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

warnings.filterwarnings("ignore")


def main():
    BookingFile = "book/exp.csv"
    # BookingFile = args[1]
    HoldFile = "data/hold.csv"
    MainLampFile = "data/mainlamp.csv"
    BackMainLampFile = "data/back_mainlamp.csv"
    AfrMainLampFile = "data/afr_mainlamp.csv"
    AfrFile = "data/accepted_filling_rate.csv"
    StressFile = "data/stress_mainlamp.csv"

    # print("File:" + BookingFile)
    booking_name = BookingFile.split("/")[1].split(".")[0]
    # 注文情報の読み込み
    T, L, D, J, U, A, G, J_small, J_medium, J_large, Port, Check_port, Booking, divide_dic\
        = read_booking.Read_booking(BookingFile)

    # 船体情報の読み込み1
    I, B, I_pair, I_next, I_same, I_lamp, I_deck, RT_benefit, delta_s, min_s, max_s, delta_h, max_h, Hold_encode, Hold\
        = read_hold.Read_hold(HoldFile)
        
    filling_rate = read_hold.Read_other(AfrFile)
    
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

        
    def assign_to_hold(assignment_list):
        hold_assignment = []
        for i in range(HOLD_COUNT):
            hold_assignment.append([])
        unloaded_orders = []

        half_way_assignments = [] #1港目を積み終えたときのバランスを計算するための配列
        for loading_port_num in range(len(L)-1):     
            tmp = {}
            for hold_num in range(HOLD_COUNT):
                tmp[hold_num] = []
            half_way_assignments.append(tmp)
        
        balance_penalty = 0

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
                            possible_unit_cnt = int(left_spaces[hold] // assignment_RT[loading_port_num][order_cnt])
                            if (possible_unit_cnt>0):
                                hold_assignment[hold].append([assignment[loading_port_num][order_cnt],possible_unit_cnt])
                                assignment_total_space[loading_port_num][order_cnt] -= assignment_RT[loading_port_num][order_cnt] * possible_unit_cnt
                                assignment_unit[loading_port_num][order_cnt] -= possible_unit_cnt
                    
                else: #最後に積む港ではない場合
                    for hold in segment:
                        ALLOWANCE_SPACE = B[hold] *  (1-filling_rate[hold])
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
                            possible_unit_cnt = int((left_spaces[hold]-ALLOWANCE_SPACE) // assignment_RT[loading_port_num][order_cnt])
                            if (possible_unit_cnt>0):
                                hold_assignment[hold].append([assignment[loading_port_num][order_cnt],possible_unit_cnt])
                                half_way_assignments[loading_port_num][hold].append([assignment[loading_port_num][order_cnt],possible_unit_cnt])
                                assignment_total_space[loading_port_num][order_cnt] -= assignment_RT[loading_port_num][order_cnt] * possible_unit_cnt
                                assignment_unit[loading_port_num][order_cnt] -= possible_unit_cnt
 
                for index in range(len(assignment_unit[loading_port_num])): 
                    if assignment_unit[loading_port_num][index]>0:
                        unloaded_orders.append([orders[index],assignment_unit[loading_port_num][index]])


        # バランス制約を計算
        # 最後に積む港以外
        balance1 = 0
        balance2 = 0
        balance3 = 0
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
                        balance3 += delta_s[hold_num] * G[order[0]] * order[1]
                        # balance_penalty += max(0,min_s-(delta_s[hold_num] * G[order[0]] * order[1]))
        balance_penalty += max(0,balance1-max_h)
        balance_penalty += max(0,balance2-max_s)
        balance_penalty += max(0,min_s-balance3)
        
        balance1 = 0
        balance2 = 0
        balance3 = 0
                
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
                    balance3 += delta_s[hold_num] * G[order[0]] * order[1]
                    # balance_penalty += max(0,min_s-(delta_s[hold_num] * G[order[0]] * order[1]))        
        balance_penalty += max(0,balance1-max_h)
        balance_penalty += max(0,balance2-max_s)
        balance_penalty += max(0,min_s-balance3)
        """
        返り値は，2次元の配列
        hold_assignment[i]で，ホールドiに割り当てられる注文の情報の配列を取得できる
        hold_assignment[i][j]で，ホールドiにj番目に割り当てられる注文を取得できる
        配列の0番目が，注文の番号 
        配列の1番目が，割り当てる台数
        """

        return hold_assignment,unloaded_orders,balance_penalty
    
    def evaluate(assignment_hold,unloaded_orders,balance_penalty):
        # 全注文内の自動車の台数を全て割り当てる
        total_left_RT = 0
        for hold_num in range(HOLD_COUNT):
            assignment = assignment_hold[hold_num]
            left_RT = B[hold_num]
            for order in assignment:
                left_RT -= A[order[0]]*order[1]
            total_left_RT += left_RT
        unloaded_units = 0
        for order in unloaded_orders:
            unloaded_units += order[1]
        # ここまで
        
        constraint1 = 0 #移動経路制約
        
        # 経路確保とバランス制約の計算に使う配列を作成
        for hold_num in range(len(assignment_hold)):         
            assignment_in_hold = assignment_hold[hold_num]
            destination_assignments = [[] for i in range(len(D))]
            for assign in assignment_in_hold:
                destination_port = int(Booking.at[assign[0],"DPORT"])
                if (destination_port-len(L)!=0):
                    for i in range(1,destination_port-len(L)+1):
                        destination_assignments[i].append(assign)  
                                   
            #降ろし地での経路確保  
            hold_space = B[hold_num]
            for i in range(1,len(destination_assignments)):
                total_loaded_space = 0
                for assign in destination_assignments[i]:
                    total_loaded_space += A[assign[0]]*assign[1]
                if total_loaded_space > hold_space*filling_rate[hold_num]:
                    constraint1 += (total_loaded_space-(hold_space*filling_rate[hold_num]))
            #ここまで
            
            #降ろし地でのバランス制約
            for i in range(1,len(destination_assignments)):
                balance1 = 0
                balance2 = 0
                balance3 = 0
                for assign in destination_assignments[i]:
                    balance1 += delta_h[hold_num] * G[assign[0]] * assign[1]
                    balance2 += delta_s[hold_num] * G[assign[0]] * assign[1]
                    balance3 += delta_s[hold_num] * G[assign[0]] * assign[1]
                balance_penalty += max(0,balance1-max_h)
                balance_penalty += max(0,balance2-max_s)
                balance_penalty += max(0,min_s-balance3)
            #ここまで
        
        objective1 = 0
        # 目的関数1 ひとつのホールドで，異なる降ろし地の注文を少なくする
        for each_assignment in assignment_hold:
            different_destination_area_orders = []
            for order in each_assignment:
                different_destination_area_orders.append(Booking.at[order[0],"DPORT"])
            unique_destination_ports = set(different_destination_area_orders)
            if (len(unique_destination_ports)>1):
                objective1 += len(unique_destination_ports)-1
        # ここまで
        return total_left_RT+unloaded_units+balance_penalty+constraint1+objective1
    
    
    

    random.seed(1)

    SEGMENT_COUNT = 18
    HOLD_COUNT = 43
    ORDER_COUNT = len(J)
    
    
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
    
    dt1 = datetime.datetime.now()
    
    # 解の初期化
    assignment = []
    # とりあえず空で初期化
    for i in range(SEGMENT_COUNT):
        tmp = []
        for j in range(len(L)): #積み地の数だけ空配列を追加
            tmp.append([])
        assignment.append(tmp)
    
    
    for i in range(len(L)):
        randomed_J = random.sample(J_t_load[i], len(J_t_load[i]))
        for j in range(len(randomed_J)):
            assignment[j%SEGMENT_COUNT][i].append(randomed_J[j])

    #初期解を，ホールドに割当
    assignment_hold,unloaded_orders,balance_penalty = assign_to_hold(assignment)
    #初期解のペナルティ    
    penalty = evaluate(assignment_hold,unloaded_orders,balance_penalty)
    shift_neighbor_list = operation.create_shift_neighbor(ORDER_COUNT,SEGMENT_COUNT)
    shift_count = 0
    
    swap_neighbor_list = operation.create_swap_neighbor(J_t_load)
    swap_count = 0
    total_improve = 1
    while total_improve != 0:
            
        while(shift_count < len(shift_neighbor_list)):
            shift_order = shift_neighbor_list[shift_count][0]
            shift_seg = shift_neighbor_list[shift_count][1]
            copied_assignment = copy.deepcopy(assignment)
            tmp_assignment,is_changed= operation.shift(copied_assignment,shift_order,shift_seg,operation.find_loading_port(shift_order,J_t_load))
            if is_changed:
                assignment_hold,unloaded_orders,balance_penalty = assign_to_hold(tmp_assignment)
                tmp_penalty = evaluate(assignment_hold,unloaded_orders,balance_penalty)
                if  tmp_penalty < penalty:
                    print("改善 "+str(tmp_penalty))
                    penalty= tmp_penalty
                    assignment = copy.deepcopy(tmp_assignment)
                    # 探索リストを最初からやり直し
                    shift_count = 0 
                    random.shuffle(shift_neighbor_list)
                else:
                    shift_count += 1
            else:
                shift_count += 1
                
        total_improve = 0
        
        while(swap_count < len(swap_neighbor_list)):
            swap_order1 = swap_neighbor_list[swap_count][0]
            swap_order2 = swap_neighbor_list[swap_count][1]
            copied_assignment = copy.deepcopy(assignment)
            tmp_assignment,is_changed = operation.swap(copied_assignment,swap_order1,swap_order2,operation.find_loading_port(swap_order1,J_t_load))
            if is_changed:
                assignment_hold,unloaded_orders,balance_penalty = assign_to_hold(tmp_assignment)
                tmp_penalty = evaluate(assignment_hold,unloaded_orders,balance_penalty)
                if  tmp_penalty < penalty:
                    print("改善 "+str(tmp_penalty))
                    penalty= tmp_penalty
                    assignment = copy.deepcopy(tmp_assignment)
                    # 探索リストを最初からやり直し
                    swap_count = 0 
                    random.shuffle(swap_neighbor_list)
                    total_improve += 1
                else:
                    swap_count += 1
            else:
                swap_count += 1
        
    assignment_hold,unloaded_orders,balance_penalty = assign_to_hold(assignment)
    penalty = evaluate(assignment_hold,unloaded_orders,balance_penalty)
    print(penalty)
    
    dt2 = datetime.datetime.now()
    print("計算時間: "+str((dt2-dt1).total_seconds())+"秒")
    
    
    result = [["Hold_ID","Order_ID","Load_Units"]]
    for index in range(len(assignment_hold)):
        assignment_dict = {}
        each_assignment = assignment_hold[index]
        for item in each_assignment:
            original_order_num = int(Booking.at[item[0],"Order_num"])
            if original_order_num in assignment_dict:
                assignment_dict[original_order_num] += item[1]
            else:
                assignment_dict[original_order_num] = item[1]
        hold_id = int(Hold_encode[Hold_encode["Index"]==index]["Hold"])
        for order_id, load_unit in assignment_dict.items():
            result.append([hold_id,order_id,load_unit])
    out_df = pd.DataFrame(result)
    out_df.to_csv("out/test.csv",index=False,header=False)            
        
if __name__ == "__main__":
    main()
