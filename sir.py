import copy
import random
import numpy as np

def count_node(G):
    s_num = sum(1 for _, d in G.nodes(data=True) if d['status'] == 'S')
    i_num = sum(1 for _, d in G.nodes(data=True) if d['status'] == 'I')
    r_num = sum(1 for _, d in G.nodes(data=True) if d['status'] == 'R')
    return s_num, i_num, r_num

def SIR_network_R(G_, source, beta, gamma):
    G = copy.deepcopy(G_)
    for n in G:
        G.nodes[n]['status'] = 'S'
    for s in source:
        G.nodes[s]['status'] = 'I'
    
    while True:
        new_inf, new_rec = [], []
        
        Inffre = {}  
        for node in G.nodes():
            if G.nodes[node]['status'] == 'S':  
        
                for nei in G.neighbors(node):
                    if G.nodes[nei]['status'] == 'I': 
             
                        if node in Inffre:
                            Inffre[node] += 1
                        else:
                            Inffre[node] = 1

        for s_node, inf_count in Inffre.items():
            infect_prob = 1 - (1 - beta) ** inf_count
            if random.uniform(0, 1) < infect_prob:
                new_inf.append(s_node)
        
        for node in G.nodes():
            if G.nodes[node]['status'] == 'I' and random.uniform(0, 1) < gamma:
                new_rec.append(node)

        for n in new_inf:
            G.nodes[n]['status'] = 'I'
        for n in new_rec:
            G.nodes[n]['status'] = 'R'
        _, i_cnt, _ = count_node(G)
        if i_cnt == 0:
            break
    
    _, _, r_cnt = count_node(G)
    return r_cnt / G.number_of_nodes()


def run_SIR_experiment_R(G, seeds, beta, gamma, repeats):
    results = []
    for _ in range(repeats):
        results.append(SIR_network_R(G, seeds, beta, gamma))
    
    mean_R = np.mean(results)
    std_R = np.std(results)
    return mean_R, std_R