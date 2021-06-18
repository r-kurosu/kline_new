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

warnings.filterwarnings("ignore")


def main():
    BookingFile = "book/exp.csv"
    # BookingFile = args[1]
    HoldFile = "data/hold.csv"
    MainLampFile = "data/mainlamp.csv"
    BackMainLampFile = "data/back_mainlamp.csv"
    AfrMainLampFile = "data/afr_mainlamp.csv"
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
            hold_assignment.append(0)
        unloaded_orders = []
        tmp = 0
        for segment_num in range(SEGMENT_COUNT):
            segment = segments[segment_num]
            assignment = assignment_list[segment_num]
            
            # 積み港ごとに分かれているものを1次元化
            assignment = list(itertools.chain.from_iterable(assignment))
            
            assignment_RT = []
            assignment_unit = []
            assignment_total_space = []
            #assignment_total_space: そのセグメントに割当てられた注文の，台数×サイズ
            
            for order in assignment:
                assignment_RT.append(A[order])
                assignment_unit.append(int(U[order]))
                assignment_total_space.append(A[order]*int(U[order]))
            # tmp += sum(assignment_unit)
            assignment_size = len(assignment)
            assignment_cnt = 0
            for hold in segment:
                space_left = B[hold]
                assignment_in_hold = []
                # 全部詰め切るか，そのホールドに注文をまるごと詰め込めなくなったらwhileを抜ける
                while (assignment_cnt<assignment_size and assignment_total_space[assignment_cnt] < space_left):
                    space_left -= assignment_total_space[assignment_cnt]
                    assignment_in_hold.append([assignment[assignment_cnt],assignment_unit[assignment_cnt]])
                    # tmp += assignment_unit[assignment_cnt]
                    assignment_total_space[assignment_cnt] = 0
                    assignment_unit[assignment_cnt] = 0
                    assignment_cnt += 1
                    
                # まるごとは注文を詰め込めなくても，一部なら可能なら一部を詰め込む
                if assignment_cnt < assignment_size:
                    possible_unit_cnt = int(space_left // assignment_RT[assignment_cnt])
                    assignment_in_hold.append([assignment[assignment_cnt],possible_unit_cnt])
                    assignment_total_space[assignment_cnt] -= assignment_RT[assignment_cnt] * possible_unit_cnt
                    assignment_unit[assignment_cnt] -= possible_unit_cnt
                    # tmp += possible_unit_cnt
                
                hold_assignment[hold] = assignment_in_hold
            # tmp += sum(assignment_unit)
            # print("-----")
            
            # 積み残しを出力する
            for index in range(len(assignment_unit)):
                if assignment_unit[index] > 0:
                    unloaded_orders.append([assignment[index],assignment_unit[index]])

        """
        返り値は，2次元の配列
        hold_assignment[i]で，ホールドiに割り当てられる注文の情報の配列を取得できる
        hold_assignment[i][j]で，ホールドiにj番目に割り当てられる注文を取得できる
        配列の0番目が，注文の番号 
        配列の1番目が，割り当てる台数
        """
        # print(tmp)
        return hold_assignment
    
    def evaluate(assignment_hold):
        total_left_RT = 0
        for hold_num in range(HOLD_COUNT):
            assignment = assignment_hold[hold_num]
            left_RT = B[hold_num]
            for order in assignment:
                left_RT -= A[order[0]]*order[1]
            total_left_RT += left_RT
        return total_left_RT
        
    """
    def evaluate(assignment_list):
        # print("evaluate")
        total_unassigned_space = 0
        for segment_num in range(SEGMENT_COUNT):
            segment = segments[segment_num]
            assignment = assignment_list[segment_num]
            # 積み港ごとに分かれているものを1次元化
            assignment = list(itertools.chain.from_iterable(assignment))
            assignment_RT = []
            assignment_unit = []
            assignment_total_space = []
            #assignment_total_space: そのセグメントに割当てられた注文の，台数×サイズ
            for order in assignment:
                assignment_RT.append(A[order])
                assignment_unit.append(int(U[order]))
                assignment_total_space.append(A[order]*int(U[order]))
            # print(assignment_total_space)
            #注文情報を揃えた
            
            #ホールドに，入る限り注文をたくさんつめこんでいく
            assignment_size = len(assignment_total_space)
            assignment_cnt = 0
            for hold in segment:
                space_left = B[hold]
                # 全部詰め切るか，そのホールドに注文をまるごと詰め込めなくなったらwhileを抜ける
                while (assignment_cnt<assignment_size and assignment_total_space[assignment_cnt] < space_left):
                    space_left -= assignment_total_space[assignment_cnt]
                    assignment_total_space[assignment_cnt] = 0
                    assignment_cnt += 1
                # まるごとは注文を詰め込めなくても，一部なら可能なら一部を詰め込む
                if assignment_cnt < assignment_size:
                    possible_unit_cnt = int(space_left // assignment_RT[assignment_cnt])
                    assignment_total_space[assignment_cnt] -= assignment_RT[assignment_cnt] * possible_unit_cnt
            # print(assignment_total_space)
            total_unassigned_space += sum(assignment_total_space)
            # print("--------")
        return total_unassigned_space
    """

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
    assignment_hold = assign_to_hold(assignment)    
    #初期解のペナルティ
    for item in assignment_hold:
        print(item)
    
    penalty = evaluate(assignment_hold)
    
    shift_neighbor_list = operation.create_shift_neighbor(ORDER_COUNT,SEGMENT_COUNT)
    shift_count = 0
    
    swap_neighbor_list = operation.create_swap_neighbor(J_t_load)
    swap_count = 0
    total_improve = 1
    
    while total_improve != 0:

        while(shift_count < len(shift_neighbor_list)):
            shift_order = shift_neighbor_list[shift_count][0]
            shift_seg = shift_neighbor_list[shift_count][1]
            tmp_assignment= operation.shift(assignment,shift_order,shift_seg,operation.find_loading_port(shift_order,J_t_load))
            assignment_hold = assign_to_hold(tmp_assignment)
            tmp_penalty = evaluate(assignment_hold)
            if  tmp_penalty <= penalty:
                print("改善 "+str(tmp_penalty))
                penalty= tmp_penalty
                assignment = copy.deepcopy(tmp_assignment)
                # 探索リストを最初からやり直し
                shift_count = 0 
                random.shuffle(shift_neighbor_list)
            else:
                shift_count += 1
                
        total_improve = 0

        while(swap_count < len(swap_neighbor_list)):
            swap_order1 = swap_neighbor_list[swap_count][0]
            swap_order2 = swap_neighbor_list[swap_count][1]
            tmp_assignment = operation.swap(assignment,swap_order1,swap_order2,operation.find_loading_port(swap_order1,J_t_load))
            assignment_hold = assign_to_hold(tmp_assignment)
            tmp_penalty = evaluate(assignment_hold)
            if  tmp_penalty <= penalty:
                print("改善 "+str(tmp_penalty))
                penalty= tmp_penalty
                assignment = copy.deepcopy(tmp_assignment)
                # 探索リストを最初からやり直し
                swap_count = 0 
                random.shuffle(swap_neighbor_list)
                total_improve += 1
            else:
                swap_count += 1


    for assign in assign_to_hold(assignment)[0]:
        print(assign)
if __name__ == "__main__":
    main()
