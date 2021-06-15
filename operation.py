import random
random.seed(1)

def find_loading_port(order_num,J_t_load):
    for port_num in range(len(J_t_load)):
        if order_num in J_t_load[port_num]:
            return port_num
        
def shift(assignment,order_num,next_segment_num,loading_port):
    current_segment_num = -1
    #どのセグメントに入っているか確認
    for seg in range(len(assignment)):
        if current_segment_num != -1:
            break
        if order_num in assignment[seg][loading_port]:
            current_segment_num = seg
    
    #挿入先が変わらないならreturn
    if current_segment_num == next_segment_num:
        return assignment
    
    # いまの割当から削除
    assignment[current_segment_num][loading_port].remove(order_num)
    #新しく挿入
    random_order = random.randint(0,len(assignment[next_segment_num][loading_port]))
    assignment[next_segment_num][loading_port].insert(random_order,order_num)
    return assignment

def swap(assignment,order1_num,order2_num,loading_port):
    order1_current_seg_num = -1
    order2_current_seg_num = -1
    #どのセグメントに入っているか確認
    for seg in range(len(assignment)):
        if order1_current_seg_num != -1 and order2_current_seg_num != -1:
            break
        if order1_num in assignment[seg][loading_port]:
            order1_current_seg_num = seg
            order1_idx = assignment[seg][loading_port].index(order1_num)
        if order2_num in assignment[seg][loading_port]:
            order2_current_seg_num = seg
            order2_idx = assignment[seg][loading_port].index(order2_num)
    if order1_current_seg_num == order2_current_seg_num:
        return assignment
    
    # swap
    assignment[order1_current_seg_num][loading_port][order1_idx] = order2_num
    assignment[order2_current_seg_num][loading_port][order2_idx] = order1_num
    return assignment

def create_shift_neighbor(order_count,segment_count):
    lis = []
    for order in range(order_count):
        for segment in range(segment_count):
            lis.append([order,segment])
            
    random.shuffle(lis)
    return lis

def create_swap_neighbor(J_t_load):
    lis = []
    for orders in J_t_load:
        if orders != []:
            order_count = len(orders)
            print(order_count)
            for order1 in orders:
                for order2 in orders:
                    if order1 != order2:
                        lis.append([order1,order2])
    return lis