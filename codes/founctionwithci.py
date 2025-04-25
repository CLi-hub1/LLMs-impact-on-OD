import pandas as pd
from sklearn.cluster import DBSCAN
import AmultiHK_CHAT as ACHT
import AmultiHK_NORMAL as ANOR
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
import hdbscan
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import matplotlib.pyplot as plt
simulations = 100
def valuescollection( A, B, C, D, E, F, G):
    def compute_values(NV_CHT):
        def compute_iteration_mean(df):
            return df.mean(axis=1)
        def compute_iteration_mean_change(df):
            return df.diff().abs().mean(axis=1)
        def compute_iteration_std(df):
            return df.std(axis=1)
        def compute_initial_mean(df):
            return df.iloc[0].mean()
        def compute_final_mean(df):
            return df.iloc[-1].mean()
        def compute_clusters(df):
            num_clusters_list = []
            for index, row in df.iterrows():
                data = np.array(row)
                dist_matrix = np.zeros((len(data), len(data)))
                for i in range(len(data)):
                    for j in range(len(data)):
                        dist_matrix[i][j] = np.sum(np.abs(data[i] - data[j]))
                max_value = np.amax(np.amax(dist_matrix, axis=1))
                min_value = np.amin(np.amin(dist_matrix, axis=1))
                valuerange = max_value - min_value
                Z = linkage(dist_matrix, 'single')
                Z[Z[:, 2] > valuerange, 2] = 0
                threshold1 = 0.2
                clusters = fcluster(Z, t=threshold1, criterion='distance')
                unique_clusters = sorted(set(clusters))
                for i in unique_clusters:
                    cluster_indices = np.where(clusters == i)[0]
                    for j in cluster_indices:
                        clusters[j] = min(cluster_indices)
                clusters = [label - min(clusters) + 1 for label in clusters]
                num_clusters = len(set(clusters))
                num_clusters_list.append(num_clusters)
            clusters_df = num_clusters_list
            return clusters_df
        result = {}
        if 'NIN' in NV_CHT.columns:
            NIN_mean = compute_iteration_mean(NV_CHT[['NIN']])
            NIN_mean_change = compute_iteration_mean_change(NV_CHT[['NIN']])
            NIN_SD = compute_iteration_std(NV_CHT[['NIN']])
            NIN_initial_mean = compute_initial_mean(NV_CHT[['NIN']])
            NIN_final_mean = compute_final_mean(NV_CHT[['NIN']])
            NIN_clusters = compute_clusters(NV_CHT[['NIN']])
            result['NIN_mean'] = NIN_mean
            result['NIN_mean_change'] = NIN_mean_change
            result['NIN_SD'] = NIN_SD
            result['NIN_initial_mean'] = NIN_initial_mean
            result['NIN_final_mean'] = NIN_final_mean
            result['NIN_clusters'] = NIN_clusters

        if 'NICN' in NV_CHT.columns:
            NICN_mean = compute_iteration_mean(NV_CHT[['NICN']])
            NICN_mean_change = compute_iteration_mean_change(NV_CHT[['NICN']])
            NICN_SD = compute_iteration_std(NV_CHT[['NICN']])
            NICN_initial_mean = compute_initial_mean(NV_CHT[['NICN']])
            NICN_final_mean = compute_final_mean(NV_CHT[['NICN']])
            NICN_clusters = compute_clusters(NV_CHT[['NICN']])
            result['NICN_mean'] = NICN_mean
            result['NICN_mean_change'] = NICN_mean_change
            result['NICN_SD'] = NICN_SD
            result['NICN_initial_mean'] = NICN_initial_mean
            result['NICN_final_mean'] = NICN_final_mean
            result['NICN_clusters'] = NICN_clusters

        if 'NIC' in NV_CHT.columns:
            NIC_mean = compute_iteration_mean(NV_CHT[['NIC']])
            NIC_mean_change = compute_iteration_mean_change(NV_CHT[['NIC']])
            NIC_SD = compute_iteration_std(NV_CHT[['NIC']])
            NIC_initial_mean = compute_initial_mean(NV_CHT[['NIC']])
            NIC_final_mean = compute_final_mean(NV_CHT[['NIC']])
            NIC_clusters = compute_clusters(NV_CHT[['NIC']])
            result['NIC_mean'] = NIC_mean
            result['NIC_mean_change'] = NIC_mean_change
            result['NIC_SD'] = NIC_SD
            result['NIC_initial_mean'] = NIC_initial_mean
            result['NIC_final_mean'] = NIC_final_mean
            result['NIC_clusters'] = NIC_clusters

        all_nodes_mean = compute_iteration_mean(NV_CHT)
        all_nodes_mean_change = compute_iteration_mean_change(NV_CHT)
        all_nodes_SD = compute_iteration_std(NV_CHT)
        all_nodes_initial_mean = compute_initial_mean(NV_CHT)
        all_nodes_final_mean = compute_final_mean(NV_CHT)
        all_nodes_clusters = compute_clusters(NV_CHT)
        result['all_nodes_mean'] = all_nodes_mean
        result['all_nodes_mean_change'] = all_nodes_mean_change
        result['all_nodes_SD'] = all_nodes_SD
        result['all_nodes_initial_mean'] = all_nodes_initial_mean
        result['all_nodes_final_mean'] = all_nodes_final_mean
        result['all_nodes_clusters'] = all_nodes_clusters
        return result

    results = {}
    for i in range(simulations):
        NV_CHT = ACHT.opiniondynamic_CHT( A, B, C, D, E,F,G)
        values = compute_values(NV_CHT)
        for key, value in values.items():
            if key not in results:
                results[key] = []
            results[key].append(value)
    return results
params_list = [
    [100, 100, 0.4, 0.6, 0.2, 0.2, -1],
    [300, 100, 0.4, 0.6, 0.2, 0.2, -1],
    [100, 300, 0.4, 0.6, 0.2, 0.2, -1],
    [100, 100, 0.8, 0.6, 0.2, 0.2, -1],
    [100, 100, 0.4, 0.2, 0.6, 0.2, -1],
    [100, 100, 0.4, 0.2, 0.2, 0.6, -1],
    [100, 100, 0.4, 0.2, 0.2, 0.6, 1]
]

def compute_ci95_from_series_list(series_list):
    df = pd.concat(series_list, axis=1)       
    mean = df.mean(axis=1)                 
    ci95 = 1.96 * df.sem(axis=1)                
    return mean, ci95

df_all_result = pd.DataFrame()

for params in params_list:
    tqdm.write(f"params: {params}")
    results = valuescollection(*params)
    mean_all_nodes_mean, ci95_all_nodes_mean = compute_ci95_from_series_list(results['all_nodes_mean'])
    mean_all_nodes_mean_change, ci95_all_nodes_mean_change = compute_ci95_from_series_list(results['all_nodes_mean_change'])
    mean_all_nodes_SD, ci95_all_nodes_SD = compute_ci95_from_series_list(results['all_nodes_SD'])
    cluster_series_list = [pd.Series(v) for v in results['all_nodes_clusters']]
    mean_all_nodes_clusters, ci95_all_nodes_clusters = compute_ci95_from_series_list(cluster_series_list)
    df_result = pd.DataFrame({
        'mean': mean_all_nodes_mean,
        'ci95': ci95_all_nodes_mean,
        'change_mean': mean_all_nodes_mean_change,
        'change_ci95': ci95_all_nodes_mean_change,
        'std_mean': mean_all_nodes_SD,
        'std_ci95': ci95_all_nodes_SD,
        'cluster_mean': mean_all_nodes_clusters,
        'cluster_ci95': ci95_all_nodes_clusters
    })
    df_result['params'] = str(params)

    df_all_result = pd.concat([df_all_result, df_result])
    df_all_result.to_csv(r'fig3withLLM.csv', index=False)

df_all_result.to_csv(r'fig3withLLM_results.csv', index=False)
def valuescollection_G1( A, B, C, G):
    def compution(NV_NOR):
        def compute_iteration_mean(df):
            return df.mean(axis=1)
        def compute_iteration_mean_change(df):
            return df.diff().abs().mean(axis=1)
        def compute_iteration_std(df):
            return df.std(axis=1)
        def compute_initial_mean(df):
            return df.iloc[0].mean()
        def compute_final_mean(df):
            return df.iloc[-1].mean()
        def compute_clusters(df):
            num_clusters_list = []
            for index, row in df.iterrows():
                data = np.array(row)
                dist_matrix = np.zeros((len(data), len(data)))
                for i in range(len(data)):
                    for j in range(len(data)):
                        dist_matrix[i][j] = np.sum(np.abs(data[i] - data[j]))
                max_value = np.amax(np.amax(dist_matrix, axis=1))
                min_value = np.amin(np.amin(dist_matrix, axis=1))
                valuerange = max_value - min_value
                Z = linkage(dist_matrix, 'single')
                Z[Z[:, 2] > valuerange, 2] = 0
                threshold1 = 0.2
                clusters = fcluster(Z, t=threshold1, criterion='distance')
                unique_clusters = sorted(set(clusters))
                for i in unique_clusters:
                    cluster_indices = np.where(clusters == i)[0]
                    for j in cluster_indices:
                        clusters[j] = min(cluster_indices)
                clusters = [label - min(clusters) + 1 for label in clusters]
                num_clusters = len(set(clusters))
                num_clusters_list.append(num_clusters)
            clusters_df =  num_clusters_list 
            return clusters_df
        result = {}
        origin_mean = compute_iteration_mean(NV_NOR[['NIN']])
        origin_mean_change = compute_iteration_mean_change(NV_NOR[['NIN']])
        origin_SD = compute_iteration_std(NV_NOR[['NIN']])
        origin_initial_mean = compute_initial_mean(NV_NOR[['NIN']])
        origin_final_mean = compute_final_mean(NV_NOR[['NIN']])
        origin_clusters = compute_clusters(NV_NOR[['NIN']])
        result['origin_mean'] = origin_mean
        result['origin_mean_change'] = origin_mean_change
        result['origin_SD'] = origin_SD
        result['origin_initial_mean'] = origin_initial_mean
        result['origin_final_mean'] = origin_final_mean
        result['origin_clusters'] = origin_clusters
        return result

    results = {}
    for i in range(simulations):
        NV_NOR = ANOR.opiniondynamic_normal(A, B, C, G)
        values = compution(NV_NOR)
        for key, value in values.items():
            if key not in results:
                results[key] = []
            results[key].append(value)
    return results

def compute_ci95_from_series_list(series_list):
    df = pd.concat(series_list, axis=1) 
    mean = df.mean(axis=1)          
    ci95 = 1.96 * df.sem(axis=1)       
    return mean, ci95

results1 = valuescollection_G1(100, 100, 0.4, 1)
mean_origin_mean, ci95_origin_mean = compute_ci95_from_series_list(results1['origin_mean'])
mean_origin_mean_change, ci95_origin_mean_change = compute_ci95_from_series_list(results1['origin_mean_change'])
mean_origin_SD, ci95_origin_SD = compute_ci95_from_series_list(results1['origin_SD'])
cluster_series_list = [pd.Series(v) for v in results1['origin_clusters']]
mean_origin_clusters, ci95_origin_clusters = compute_ci95_from_series_list(cluster_series_list)
df_origin_result = pd.DataFrame({
    'mean': mean_origin_mean,
    'ci95': ci95_origin_mean,
    'change_mean': mean_origin_mean_change,
    'change_ci95': ci95_origin_mean_change,
    'std_mean': mean_origin_SD,
    'std_ci95': ci95_origin_SD,
    'cluster_mean': mean_origin_clusters,
    'cluster_ci95': ci95_origin_clusters
})

df_origin_result.to_csv(r'fig3G1_withoutLLM.csv', index=False)
import matplotlib.pyplot as plt
import pandas as pd

withLLM = pd.read_csv(r'fig3withLLM_results.csv')
withoutLLM = pd.read_csv(r'fig3G1_withoutLLM.csv')

params_aa = [100, 100, 0.4, 0.6, 0.2, 0.2, -1]
params_bb = [300, 100, 0.4, 0.6, 0.2, 0.2, -1]
params_cc = [100, 300, 0.4, 0.6, 0.2, 0.2, -1]
params_dd = [100, 100, 0.8, 0.6, 0.2, 0.2, -1]
params_ee = [100, 100, 0.4, 0.2, 0.6, 0.2, -1]
params_ff = [100, 100, 0.4, 0.2, 0.2, 0.6, -1]
params_gg = [100, 100, 0.4, 0.2, 0.2, 0.6, 1]

withLLM['params'] = withLLM['params'].apply(lambda x: eval(x) if isinstance(x, str) else x)

df_aa = withLLM[withLLM['params'].apply(lambda x: x == params_aa)].reset_index(drop=True)
df_bb = withLLM[withLLM['params'].apply(lambda x: x == params_bb)].reset_index(drop=True)
df_cc = withLLM[withLLM['params'].apply(lambda x: x == params_cc)].reset_index(drop=True)
df_dd = withLLM[withLLM['params'].apply(lambda x: x == params_dd)].reset_index(drop=True)
df_ee = withLLM[withLLM['params'].apply(lambda x: x == params_ee)].reset_index(drop=True)
df_ff = withLLM[withLLM['params'].apply(lambda x: x == params_ff)].reset_index(drop=True)
df_gg = withLLM[withLLM['params'].apply(lambda x: x == params_gg)].reset_index(drop=True)

cloumn_names = ['mean', 'ci95', 'change_mean', 'change_ci95', 'std_mean', 'std_ci95', 'cluster_mean', 'cluster_ci95']
df_list = [df_aa, df_bb, df_cc, df_dd, df_ee, df_ff, df_gg]
df_labels = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg']
plt.rcParams['font.family'] = 'Arial'
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), dpi = 600)
colors = ['orange', 'purple', 'blue', 'green', 'red', 'teal', 'magenta']
markers = ['*', 'd', 'o', '^', '2', 'x', 's']
columns_for_subplots = [
    ('mean', 'ci95'),
    ('change_mean', 'change_ci95'),
    ('std_mean', 'std_ci95'),
    ('cluster_mean', 'cluster_ci95')
]

for idx, (row, col) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
    ax = axes[row, col]
    val_col, ci_col = columns_for_subplots[idx]
    ax.plot(withoutLLM.index, withoutLLM[val_col], linestyle='-', marker='+', markersize=6,
            color='black') 
    ax.fill_between(withoutLLM.index,
                    withoutLLM[val_col] - withoutLLM[ci_col],
                    withoutLLM[val_col] + withoutLLM[ci_col],
                    color='black', alpha=0.2)
    for i, df in enumerate(df_list):
        if val_col in df.columns and ci_col in df.columns:
            ax.plot(df.index, df[val_col], linestyle='-', marker=markers[i], markersize=4,
                    color=colors[i]) 
            ax.fill_between(df.index,
                            df[val_col] - df[ci_col],
                            df[val_col] + df[ci_col],
                            color=colors[i], alpha=0.2)
    ax.set_xlim([0, 100])
    ax.set_xlabel('Iteration', fontdict={'color': 'black','weight': 'bold','size': 17})
    if val_col == 'mean':
        ax.set_ylabel('Mean opinion value', fontdict={'color': 'black','weight': 'bold','size': 17})
    elif val_col == 'change_mean':
        ax.set_ylabel('Mean opinion value change', fontdict={'color': 'black','weight': 'bold','size': 17})
    elif val_col == 'std_mean':
        ax.set_ylabel('Mean standard deviation', fontdict={'color': 'black','weight': 'bold','size': 17})
    elif val_col == 'cluster_mean':
        ax.set_ylabel('Mean number of clusters', fontdict={'color': 'black','weight': 'bold','size': 17})

plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.tight_layout()
plt.show()
