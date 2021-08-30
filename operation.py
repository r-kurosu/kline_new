import random
random.seed(1)

def find_loading_port(order_num,J_t_load):
    for port_num in range(len(J_t_load)):
        if order_num in J_t_load[port_num]:
            return port_num
        
def find_current_segment_and_index(assignment,order_num,loading_port):
    for seg in range(len(assignment)):
        if order_num in assignment[seg][loading_port]:
            idx = assignment[seg][loading_port].index(order_num)
            return seg,idx

            
def shift(assignment,order_num,current_segment_num,next_segment_num,loading_port,next_order):
    # いまの割当から削除
    assignment[current_segment_num][loading_port].remove(order_num)
    #新しく挿入
    assignment[next_segment_num][loading_port].insert(next_order,order_num)
    return assignment

    #挿入先が変わらないならreturn
    if current_segment_num == next_segment_num:
        assignment[current_segment_num][loading_port].remove(order_num)
        assignment[next_segment_num][loading_port].append(order_num)
        return assignment,True
    else:
        assignment[current_segment_num][loading_port].remove(order_num)
        assignment[next_segment_num][loading_port].append(order_num)
        return assignment,True
        # いまの割当から削除
        assignment[current_segment_num][loading_port].remove(order_num)
        #新しく挿入
        random_order = random.randint(0,len(assignment[next_segment_num][loading_port]))
        assignment[next_segment_num][loading_port].insert(random_order,order_num)
        # assignment[next_segment_num][loading_port].append(order_num)
        return assignment,True

def swap(assignment,order1_num,order1_current_segment,order1_idx,order2_num,order2_current_segment,order2_idx,loading_port):
    assignment[order1_current_segment][loading_port][order1_idx] = order2_num
    assignment[order2_current_segment][loading_port][order2_idx] = order1_num
    return assignment

def intra(assignment,segment_num,loading_port_num):
    first_idx = random.randint(0,len(assignment[segment_num][loading_port_num])-1)
    next_idx = random.randint(0,len(assignment[segment_num][loading_port_num])-1)
    while first_idx == next_idx:
        next_idx = random.randint(0,len(assignment[segment_num][loading_port_num])-1)
    assignment[segment_num][loading_port_num][first_idx],assignment[segment_num][loading_port_num][next_idx] = assignment[segment_num][loading_port_num][next_idx],assignment[segment_num][loading_port_num][first_idx]
    
    return assignment

def create_shift_neighbor(order_count,segment_count):
    lis = []
    for order in range(order_count):
        for segment in range(segment_count):
            lis.append([order,segment])
            
    random.shuffle(lis)
    return lis

def create_swap_neighbor(J_t_load,Booking):
    lis = []
    for orders in J_t_load:
        if orders != []:
            order_count = len(orders)
            for i in range(0,order_count-1):
                for j in range(i+1,order_count):
                    print(orders[i],orders[j])
                    lis.append([orders[i],orders[j]])
    random.shuffle(lis)
    return lis