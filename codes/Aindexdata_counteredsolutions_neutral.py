import pandas as pd
from sklearn.cluster import DBSCAN
import AmultiHK_CHAT_neutral as ACHT
import AmultiHK_NORMAL as ANOR
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
import hdbscan
from sklearn.metrics.pairwise import euclidean_distances


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
            data = np.array(df.iloc[-1])
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
            return num_clusters
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
            data = np.array(df.iloc[-1])
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
            return num_clusters
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

