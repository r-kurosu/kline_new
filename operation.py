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

def create_shift_neighbor(order_count,segment_count):
    print(order_count)
    print(segment_count)
    lis = []
    for order in range(order_count):
        for segment in range(segment_count):
            lis.append([order,segment])
    random.shuffle(lis)
    return random