import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
import random
import numpy as np
import pandas as pd
from Aindexdata_counteredsolutions_neutral import valuescollection,valuescollection_G1
import time 
from tqdm import tqdm
import itertools
from multiprocessing import Pool

# simulations = 100

def values_all(A, B, C, D, E, F, G):
    results = valuescollection(A, B, C, D, E, F, G)
    # 将每个变量的n次重复计算结果合并为一个 DataFrame
    def find_first_small_col(df):
        if df.empty:
            return None
        for i, col in enumerate(df.columns):
            if (df.loc[:, col] < 0.005).all():
                return i
        return df.shape[1]-1
    list2 = []
    list2 = ['NIN_mean_change_col_idx','NICN_mean_change_col_idx','NIC_mean_change_col_idx','all_nodes_mean_change_col_idx']
    for result in list2:
        globals()[f"{result}"] = 0
    
    def clusters(df):
        if df.empty:
            return None
        else:
            return float((df.mean()).item())
    list3 = []
    list3 = ['NIN_clusters','NICN_clusters','NIC_clusters','all_clusters']
    for result in list3:
        globals()[f"{result}"] = 0

    if 'NIN_mean' in results:
        df_NIN_mean = pd.DataFrame(results['NIN_mean'])
        df_NIN_mean_change = pd.DataFrame(results['NIN_mean_change'])
        df_NIN_SD = pd.DataFrame(results['NIN_SD'])
        df_NIN_initial_mean = pd.DataFrame(results['NIN_initial_mean'])
        df_NIN_final_mean = pd.DataFrame(results['NIN_final_mean'])
        df_NIN_clusters = pd.DataFrame(results['NIN_clusters'])
        NIN_mean_diff = df_NIN_final_mean.mean().iloc[0] - df_NIN_initial_mean.mean().iloc[0]
        NIN_mean_change_col_idx = find_first_small_col(df_NIN_mean_change)
        NIN_finalSD = df_NIN_SD.iloc[:,-1].mean()
        NIN_clusters = clusters(df_NIN_clusters)
    else:
        NIN_mean_diff = None
        NIN_mean_change_col_idx = None
        NIN_finalSD = None
        NIN_clusters = None

    if 'NICN_mean' in results:    
        df_NICN_mean = pd.DataFrame(results['NICN_mean'])
        df_NICN_mean_change = pd.DataFrame(results['NICN_mean_change'])
        df_NICN_SD = pd.DataFrame(results['NICN_SD'])
        df_NICN_initial_mean = pd.DataFrame(results['NICN_initial_mean'])
        df_NICN_final_mean = pd.DataFrame(results['NICN_final_mean'])
        df_NICN_clusters = pd.DataFrame(results['NICN_clusters'])
        NICN_mean_diff = df_NICN_final_mean.mean().iloc[0] - df_NICN_initial_mean.mean().iloc[0]
        NICN_mean_change_col_idx = find_first_small_col(df_NICN_mean_change)
        NICN_finalSD = df_NICN_SD.iloc[:,-1].mean()
        NICN_clusters = clusters(df_NICN_clusters)
    else:
        NICN_mean_diff = None
        NICN_mean_change_col_idx = None
        NICN_finalSD = None
        NICN_clusters = None

    if 'NIC_mean' in results:      
        df_NIC_mean = pd.DataFrame(results['NIC_mean'])
        df_NIC_mean_change = pd.DataFrame(results['NIC_mean_change'])
        df_NIC_SD = pd.DataFrame(results['NIC_SD'])
        df_NIC_initial_mean = pd.DataFrame(results['NIC_initial_mean'])
        df_NIC_final_mean = pd.DataFrame(results['NIC_final_mean'])
        df_NIC_clusters = pd.DataFrame(results['NIC_clusters'])
        NIC_mean_diff = df_NIC_final_mean.mean().iloc[0] - df_NIC_initial_mean.mean().iloc[0]
        NIC_mean_change_col_idx = find_first_small_col(df_NIC_mean_change)
        NIC_finalSD = df_NIC_SD.iloc[:,-1].mean()
        NIC_clusters = clusters(df_NIC_clusters)
    else:
        NIC_mean_diff = None
        NIC_mean_change_col_idx = None
        NIC_finalSD = None
        NIC_clusters = None

    if 'all_nodes_mean' in results:       
        df_all_nodes_mean = pd.DataFrame(results['all_nodes_mean'])
        df_all_nodes_mean_change = pd.DataFrame(results['all_nodes_mean_change'])
        df_all_nodes_SD = pd.DataFrame(results['all_nodes_SD'])
        df_all_nodes_initial_mean = pd.DataFrame(results['all_nodes_initial_mean'])
        df_all_nodes_final_mean = pd.DataFrame(results['all_nodes_final_mean'])
        df_all_nodes_clusters = pd.DataFrame(results['all_nodes_clusters'])
        all_nodes_mean_diff = df_all_nodes_final_mean.mean().iloc[0] - df_all_nodes_initial_mean.mean().iloc[0]
        all_nodes_mean_change_col_idx = find_first_small_col(df_all_nodes_mean_change)
        all_finalSD = df_all_nodes_SD.iloc[:,-1].mean()
        all_clusters = clusters(df_all_nodes_clusters)
    else:
        all_nodes_mean_diff = None
        all_nodes_mean_change_col_idx = None
        all_finalSD = None
        all_clusters = None

    return NIN_mean_diff,NICN_mean_diff,NIC_mean_diff,all_nodes_mean_diff,NIN_mean_change_col_idx,NICN_mean_change_col_idx,NIC_mean_change_col_idx,all_nodes_mean_change_col_idx,NIN_finalSD,NICN_finalSD,NIC_finalSD,all_finalSD,NIN_clusters,NICN_clusters,NIC_clusters,all_clusters


# NIN_mean_diff,NICN_mean_diff,NIC_mean_diff,all_nodes_mean_diff,NIN_mean_change_col_idx,NICN_mean_change_col_idx,NIC_mean_change_col_idx,all_nodes_mean_change_col_idx,NIN_finalSD,NICN_finalSD,NIC_finalSD,all_finalSD,NIN_clusters,NICN_clusters,NIC_clusters,all_clusters = values_all(100, 100, 0.4, 0.6, 0.2, 0.2, -1)

# 初始化一个DataFrame，用于存储所有的计算结果
results_df = pd.DataFrame(columns=['nodenum', 'iteration', 'threshold', 'per_NIN', 'per_NINL', 'per_NIL', 'LLM_VALUE', 'NIN_mean_diff', 'NINL_mean_diff', 'NIL_mean_diff', 'all_mean_diff', 
                                   'NIN_converge_times', 'NINL_converge_times', 'NIL_converge_times', 'all_converge_times', 'NIN_finalSD','NINL_finalSD','NIL_finalSD','all_finalSD','NIN_clusters','NINL_clusters','NIL_clusters','all_clusters'])

# 遍历所有参数组合
A = 100
B = 100
G = -1
for C in [i/10 for i in range(11)]:
    for D in [i/10 for i in range(11)]:
        for E in [i/10 for i in range(11-int(10*D))]:
            # 确保B+C+D=1
            F = round(1 - D - E, 1)
            tqdm.write(f"params: {A,B,C,D,E,F,G}")
            NIN_mean_diff, NICN_mean_diff, NIC_mean_diff, all_mean_diff, NIN_converge_times, NICN_converge_times, NIC_converge_times, all_converge_times, NIN_finalSD,NICN_finalSD,NIC_finalSD,all_finalSD,NIN_clusters,NICN_clusters,NIC_clusters,all_clusters = values_all(A, B, C, D, E, F, G)
            # 将计算结果存储在DataFrame中
            results_df = results_df.append({'nodenum':A, 'iteration':B,'threshold': C, 'per_NIN': D, 'per_NINL': E, 'per_NIL': F, 'LLM_VALUE': G, 'NIN_mean_diff': NIN_mean_diff, 'NINL_mean_diff': NICN_mean_diff, 'NIL_mean_diff': NIC_mean_diff, 'all_mean_diff': all_mean_diff,
                                            'NIN_converge_times': NIN_converge_times, 'NINL_converge_times': NICN_converge_times, 'NIL_converge_times': NIC_converge_times, 'all_converge_times': all_converge_times,
                                            'NIN_finalSD':NIN_finalSD,'NINL_finalSD':NICN_finalSD,'NIL_finalSD':NIC_finalSD,'all_finalSD':all_finalSD,'NIN_clusters':NIN_clusters,'NINL_clusters':NICN_clusters,'NIL_clusters':NIC_clusters,'all_clusters':all_clusters}, ignore_index=True)
            # results_df.to_csv(r'C:\Users\liu\Desktop\同步\opiniondynamic\modifiedHK\countered_solutiuons\results_neutral.csv')
results_df.to_csv(r'C:\Users\liu\Desktop\同步\opiniondynamic\modifiedHK\countered_solutiuons\results_countered_neutral.csv')



# #### 改变LLM值和threshold值的情况，固定其他参数
# A = 100
# B = 100
# # C = 0.5
# D = 0.6
# E = 0.2
# F = 0.2
# G = -1
# for C in [i/10 for i in range(11)]:
#     # 调用NV_CHT生成函数，并计算节点的指标
#     tqdm.write(f"params: {A,B,C,D,E,F,G}")
#     NIN_mean_diff, NICN_mean_diff, NIC_mean_diff, all_mean_diff, NIN_converge_times, NICN_converge_times, NIC_converge_times, all_converge_times, NIN_finalSD,NICN_finalSD,NIC_finalSD,all_finalSD,NIN_clusters,NICN_clusters,NIC_clusters,all_clusters = values_all(A, B, C, D, E, F, G)
#     # 将计算结果存储在DataFrame中
#     results_df = results_df.append({'nodenum':A, 'iteration':B,'threshold': C, 'per_NIN': D, 'per_NINL': E, 'per_NIL': F, 'LLM_VALUE': G, 'NIN_mean_diff': NIN_mean_diff, 'NINL_mean_diff': NICN_mean_diff, 'NIL_mean_diff': NIC_mean_diff, 'all_mean_diff': all_mean_diff,
#                                                  'NIN_converge_times': NIN_converge_times, 'NINL_converge_times': NICN_converge_times, 'NIL_converge_times': NIC_converge_times, 'all_converge_times': all_converge_times,
#                                                  'NIN_finalSD':NIN_finalSD,'NINL_finalSD':NICN_finalSD,'NIL_finalSD':NIC_finalSD,'all_finalSD':all_finalSD,'NIN_clusters':NIN_clusters,'NINL_clusters':NICN_clusters,'NIL_clusters':NIC_clusters,'all_clusters':all_clusters}, ignore_index=True)
#     results_df.to_csv(r'D:\zju_lichao\bo1\opiniondynamic\results_countered_opponodes.csv')
# results_df.to_csv(r'D:\zju_lichao\bo1\opiniondynamic\results_countered_opponodes1.csv')


# #### 改变三种节点比例的情况，固定其他参数
# params_list = [
#     [100, 100, 0.5, 0.6, 0.2, 0.2, -1],
#     [100, 100, 0.5, 0.2, 0.6, 0.2, -1],
#     [100, 100, 0.5, 0.2, 0.2, 0.6, -1],
#     [100, 100, 0.5, 0.2, 0.2, 0.6, -1]
# ]

# for params in params_list:
#     # 调用NV_CHT生成函数，并计算节点的指标
#     tqdm.write(f"params: {params}")
#     NIN_mean_diff, NICN_mean_diff, NIC_mean_diff, all_mean_diff, NIN_converge_times, NICN_converge_times, NIC_converge_times, all_converge_times, NIN_finalSD,NICN_finalSD,NIC_finalSD,all_finalSD,NIN_clusters,NICN_clusters,NIC_clusters,all_clusters = values_all(*params)
#     # 将计算结果存储在DataFrame中
#     results_df = results_df.append({'nodenum':'A', 'iteration':'B','threshold': 'C', 'per_NIN': 'D', 'per_NINL': 'E', 'per_NIL': 'F', 'LLM_VALUE': 'G', 'NIN_mean_diff': NIN_mean_diff, 'NINL_mean_diff': NICN_mean_diff, 'NIL_mean_diff': NIC_mean_diff, 'all_mean_diff': all_mean_diff,
#                                     'NIN_converge_times': NIN_converge_times, 'NINL_converge_times': NICN_converge_times, 'NIL_converge_times': NIC_converge_times, 'all_converge_times': all_converge_times,
#                                     'NIN_finalSD':NIN_finalSD,'NINL_finalSD':NICN_finalSD,'NIL_finalSD':NIC_finalSD,'all_finalSD':all_finalSD,'NIN_clusters':NIN_clusters,'NINL_clusters':NICN_clusters,'NIL_clusters':NIC_clusters,'all_clusters':all_clusters}, ignore_index=True)
# results_df.to_csv(r'D:\zju_lichao\bo1\opiniondynamic\results_changeproportion1.csv')


####理论上能加快运行速度
# def compute_values1(params):
#     A, B, C, D, E, F, G = params
#     tqdm.write(f"params: {A,B,C,D,E,F,G}")
#     NIN_mean_diff, NICN_mean_diff, NIC_mean_diff, all_mean_diff, NIN_converge_times, NICN_converge_times, NIC_converge_times, all_converge_times, NIN_finalSD,NICN_finalSD,NIC_finalSD,all_finalSD,NIN_clusters,NICN_clusters,NIC_clusters,all_clusters = values_all(A, B, C, D, E, F, G)
#     return {'nodenum': A, 'iteration': B, 'threshold': C, 'per_NIN': D, 'per_NINL': E, 'per_NIL': F, 'LLM_VALUE': G, 'NIN_mean_diff': NIN_mean_diff, 'NINL_mean_diff': NICN_mean_diff, 'NIL_mean_diff': NIC_mean_diff, 'all_mean_diff': all_mean_diff,
#             'NIN_converge_times': NIN_converge_times, 'NINL_converge_times': NICN_converge_times, 'NIL_converge_times': NIC_converge_times, 'all_converge_times': all_converge_times,
#             'NIN_finalSD': NIN_finalSD, 'NINL_finalSD': NICN_finalSD, 'NIL_finalSD': NIC_finalSD, 'all_finalSD': all_finalSD, 'NIN_clusters': NIN_clusters, 'NINL_clusters': NICN_clusters, 'NIL_clusters': NIC_clusters, 'all_clusters': all_clusters}
# # params_list = []
# for C in [i/10 for i in range(11)]:
#     for D in [i/10 for i in range(11)]:
#         for E in [i/10 for i in range(11-int(10*D))]:
#             # 确保B+C+D=1
#             F = round(1 - D - E, 1)
#             if F < 0:
#                 continue
#             params_list.append((100, 100, C, D, E, F))

# with Pool(processes=4) as pool:
#     results = list(tqdm(pool.imap_unordered(compute_values1, params_list), total=len(params_list)))

# results_df = pd.DataFrame(results)
# results_df.to_csv(r'D:\zju_lichao\bo1\opiniondynamic\results_addSDclusters.csv')
