import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def opiniondynamic_CHT(nodenum, iterations, threshold, pro_HK, pro_compre, pro_constant, CHT):
    arbitrarynode = 1
    randnodes = 4
    pro_opposite = 0
    G = nx.erdos_renyi_graph(nodenum, 0.1) # Generate Erdos-Renyi graph

    for node in G.nodes():
        G.nodes[node]['value'] = random.uniform(-1, 1)
        G.nodes[node]['influence_degree'] = len(list(G.neighbors(node))) / (G.number_of_nodes()-1)
        G.nodes[node]['stubborn'] = random.uniform(0, 1)
    HK_nodes = random.sample(list(G.nodes()), k= int(pro_HK*nodenum))
    compre_nodes = random.sample(list(set(G.nodes()) - set(HK_nodes)), k= int(pro_compre*nodenum))
    constant_nodes = random.sample(list(set(G.nodes()) - set(HK_nodes) - set(compre_nodes)), k= int(pro_constant*nodenum))
    opposite_nodes = random.sample(list(set(G.nodes()) - set(HK_nodes) - set(compre_nodes)-set(constant_nodes)), k= int(pro_opposite*nodenum))
    
    for node in  constant_nodes:
        G.nodes[node]['value'] = CHT
    def HK_update(node, G, threshold):
        neighbors_values = [G.nodes[neighbor]['value'] for neighbor in G.neighbors(node)]
        if neighbors_values:
            diff_values = [value for value in neighbors_values if abs(G.nodes[node]['value'] - value) <= threshold]
            if diff_values:
                diff_weights = []
                for neighbor in G.neighbors(node):
                    if abs(G.nodes[node]['value'] - G.nodes[neighbor]['value']) <= threshold:
                        inf = G.nodes[node]['influence_degree']
                        diff_weights.append(inf)
                sum_weights = sum(diff_weights)
                diff_weights = [weight / sum_weights for weight in diff_weights]
                update_value = sum([diff_weights[i] * diff_values[i] for i in range(len(diff_values))])
                G.nodes[node]['value'] = update_value*(1-G.nodes[node]['stubborn'])+G.nodes[node]['value']*G.nodes[node]['stubborn']

    def compre_update(node, G, threshold):
        neighbors_values = [G.nodes[neighbor]['value'] for neighbor in G.neighbors(node)]
        if abs(G.nodes[node]['value'] - CHT) > threshold:
            if neighbors_values:
                diff_values = [value for value in neighbors_values if abs(G.nodes[node]['value'] - value) <= threshold]
                if diff_values:
                    diff_weights = []
                    for neighbor in G.neighbors(node):
                        if abs(G.nodes[node]['value'] - G.nodes[neighbor]['value']) <= threshold:
                            inf = G.nodes[node]['influence_degree']
                            diff_weights.append(inf)
                    sum_weights = sum(diff_weights)
                    diff_weights = [weight / sum_weights for weight in diff_weights]
                    update_value = sum([diff_weights[i] * diff_values[i] for i in range(len(diff_values))])
                    G.nodes[node]['value'] = update_value*(1-G.nodes[node]['stubborn'])+G.nodes[node]['value']*G.nodes[node]['stubborn']
        else:
            if neighbors_values:
                diff_values = [value for value in neighbors_values if abs(G.nodes[node]['value'] - value) <= threshold]
                if diff_values:
                    diff_weights = []
                    for neighbor in G.neighbors(node):
                        if abs(G.nodes[node]['value'] - G.nodes[neighbor]['value']) <= threshold:
                            inf = G.nodes[node]['influence_degree']
                            diff_weights.append(inf)
                    sum_weights = sum(diff_weights)
                    sum_weights = sum_weights + 1
                    diff_weights = [weight / sum_weights for weight in diff_weights]
                    update_value_neighbor = sum([diff_weights[i] * diff_values[i] for i in range(len(diff_values))])
                    update_value_AI = (1 / sum_weights) * CHT
                    update_value = update_value_neighbor + update_value_AI
                    G.nodes[node]['value'] = update_value*(1-G.nodes[node]['stubborn'])+G.nodes[node]['value']*G.nodes[node]['stubborn']

    def constant_update(node, G):
        G.nodes[node]['value'] = CHT

    def opposite_update(node, G):
        G.nodes[node]['value'] = -CHT
    
    def randomnode_update(node, G):
        G.nodes[node]['value'] = G.nodes[node]['value']   
    def generate_arbitrary_events(num, G):
        for i in range(num):
            node = max(G.nodes()) + 1  # create a new node with a unique ID
            creation = "arbitrary"
            G.add_node(node, value=random.uniform(-1, 1), stubborn=random.uniform(0, 1), influence_degree= 0, labels = creation)
            for neighbor in G.nodes():
                if neighbor != node:
                    if random.random() < G.nodes[neighbor]['influence_degree']:
                        G.add_edge(node, neighbor)
                        G.nodes[node]['influence_degree'] = len(list(G.neighbors(node))) / (G.number_of_nodes()-1)
                        G.nodes[neighbor]['influence_degree'] = (G.nodes[neighbor]['influence_degree']*G.number_of_nodes()+1) / (G.number_of_nodes()-1)

    randomnode = []
    def generate_randomnode(num, G):
        for i in range(num):
            node = max(G.nodes()) + 1  # create a new node with a unique ID
            creation = "randomnode"
            G.add_node(node,value=random.uniform(-1, 1), stubborn=random.uniform(0, 1), influence_degree= 0, labels = creation)
            for neighbor in G.nodes():
                if neighbor != node:
                    if random.random() < G.nodes[neighbor]['influence_degree']:
                        G.add_edge(node, neighbor)
                        G.nodes[node]['influence_degree'] = len(list(G.neighbors(node))) / (G.number_of_nodes()-1)
                        G.nodes[neighbor]['influence_degree'] = (G.nodes[neighbor]['influence_degree']*G.number_of_nodes()+1) / (G.number_of_nodes()-1)
            randomnode.append(node)
    node_values = []
    for i in range(iterations+1):
        if i > 0:
            generate_arbitrary_events(arbitrarynode, G)  # add arbitrary events to the graph
            arbitrary = [max(G.nodes())]
            for node in  arbitrary:
                HK_update(node, G, threshold)
            for node in HK_nodes:
                HK_update(node, G, threshold)
            for node in compre_nodes:
                compre_update(node, G, threshold)
            for node in constant_nodes:
                constant_update(node, G)
            for node in opposite_nodes:
                opposite_update(node, G)
            for node in  randomnode:
                randomnode_update(node, G)
            G.remove_node(max(G.nodes()))
            values = [G.nodes[node]['value'] for node in G.nodes()]
            if random.random() > 0.9:
                generate_randomnode(randnodes, G)
            for node in HK_nodes:
                HK_update(node, G, threshold)
            for node in compre_nodes:
                compre_update(node, G, threshold)
            for node in constant_nodes:
                constant_update(node, G)
            for node in opposite_nodes:
                opposite_update(node, G)
            for node in  randomnode:
                randomnode_update(node, G)
            values = [G.nodes[node]['value'] for node in G.nodes()]
        else:
            values = [G.nodes[node]['value'] for node in G.nodes()]
        node_values.append(values)
    df1 = pd.DataFrame(node_values)
    for col in df1.columns:
        if col in HK_nodes:
            df1 = df1.rename(columns={col: 'NIN'})
        elif col in compre_nodes:
            df1 = df1.rename(columns={col: 'NICN'})
        elif col in constant_nodes:
            df1 = df1.rename(columns={col: 'NIC'})
        elif col in randomnode:
            df1 = df1.rename(columns={col: 'NR'})
        else:
            continue
    df1 = df1.iloc[:,:nodenum]
    return df1

