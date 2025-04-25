import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
import matplotlib.pyplot as plt
import random
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

def opiniondynamic_normal(nodenum, iterations, threshold, pro_HK ):
    randnodes = 1
    G = nx.erdos_renyi_graph(nodenum, 0.1) # Generate Erdos-Renyi graph
    for node in G.nodes():
        G.nodes[node]['value'] = random.uniform(-1, 1)
        G.nodes[node]['influence_degree'] = len(list(G.neighbors(node))) / (G.number_of_nodes()-1)
        G.nodes[node]['stubborn'] = random.uniform(0, 1)

    HK_nodes = random.sample(list(G.nodes()), k= int(pro_HK*nodenum))
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

    new_nodes = []
    def generate_random_nodes(num, G):
        for i in range(num):
            node = max(G.nodes()) + 1  # create a new node with a unique ID
            creation = "new_nodes"
            G.add_node(node, value=random.uniform(-1, 1), stubborn=random.uniform(0, 1), influence_degree= 0, labels = creation)
            for neighbor in G.nodes():
                if neighbor != node:
                    if random.random() < G.nodes[neighbor]['influence_degree']:
                        G.add_edge(node, neighbor)
                        G.nodes[node]['influence_degree'] = len(list(G.neighbors(node))) / (G.number_of_nodes()-1)
                        G.nodes[neighbor]['influence_degree'] = (G.nodes[neighbor]['influence_degree']*G.number_of_nodes()+1) / (G.number_of_nodes()-1)
            new_nodes.append(node)
    node_values = []
    for i in range(iterations+1):
        if i > 0:
            generate_random_nodes(randnodes, G)  # add random nodes to the graph
            for node in  new_nodes:
                HK_update(node, G, threshold)
            for node in HK_nodes:
                HK_update(node, G, threshold)
            G.remove_node(nodenum)
            values = [G.nodes[node]['value'] for node in G.nodes()]
        else:
            values = [G.nodes[node]['value'] for node in G.nodes()]
        node_values.append(values)
    
    df2 = pd.DataFrame(node_values)
    for col in df2.columns:
        if col in HK_nodes:
            df2 = df2.rename(columns={col: 'NIN'})
        else:
            continue
    return df2

