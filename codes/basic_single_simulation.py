import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
import matplotlib.pyplot as plt
import random
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import imageio

##constant
nodenum = 100
iterations = 100
threshold = 0.5
CHT = -1
pro_HK = 0.6
pro_compre = 0.2
pro_constant = 0.2 #1- pro_HK - pro_compre

G = nx.erdos_renyi_graph(nodenum, 0.2) # Generate Erdos-Renyi graph
# G = nx.watts_strogatz_graph(nodenum, 5, 0.2, seed=10) # Returns a Watts–Strogatz small-world graph.
# G = nx.random_graphs.barabasi_albert_graph(nodenum, 3) #Scale-free network (Barabasi-Albert) 
# G = LFR_benchmark_graph(nodenum, 3, 1.5, 0.1, average_degree=5, min_community=30, seed=10)##LFR baseline graph
###G = nx.scale_free_graph(n=nodenum, seed=None) #scale-free
###G = nx.gn_graph(nodenum) #gn_graph
###some graph indicators
# degree = dict(nx.degree(G))
# connected_components = list(nx.connected_components(G))
# clustering_coefficient = nx.clustering(G)

# Initialize nodes with random values and influence degree constant
for node in G.nodes():
    G.nodes[node]['value'] = random.uniform(-1, 1)
    G.nodes[node]['influence_degree'] = len(list(G.neighbors(node))) / G.number_of_nodes()

# Specify types of nodes
HK_nodes = random.sample(list(G.nodes()), k= int(pro_HK*nodenum))
compre_nodes = random.sample(list(set(G.nodes()) - set(HK_nodes)), k= int(pro_compre*nodenum))
constant_nodes = random.sample(list(set(G.nodes()) - set(HK_nodes) - set(compre_nodes)), k= int(pro_constant*nodenum))

# Define update functions
def HK_update(node, G, threshold):
    neighbors_values = [G.nodes[neighbor]['value'] for neighbor in G.neighbors(node)]
    if neighbors_values:
        diff_values = [value for value in neighbors_values if abs(G.nodes[node]['value'] - value) <= threshold]
        if diff_values:
            diff_weights = []
            for neighbor in G.neighbors(node):
                if abs(G.nodes[node]['value'] - G.nodes[neighbor]['value']) <= threshold:
                    ##influence_degree = 1 / len(list(G.neighbors(neighbor)))
                    inf = G.nodes[node]['influence_degree']
                    diff_weights.append(inf)
            sum_weights = sum(diff_weights)
            diff_weights = [weight / sum_weights for weight in diff_weights]
            update_value = sum([diff_weights[i] * diff_values[i] for i in range(len(diff_values))])
            # G.nodes[node]['value'] = G.nodes[node]['value'] + update_value
            G.nodes[node]['value'] = update_value

def compre_update(node, G, threshold):
    neighbors_values = [G.nodes[neighbor]['value'] for neighbor in G.neighbors(node)]
    if abs(G.nodes[node]['value'] - CHT) > threshold:
        if neighbors_values:
            diff_values = [value for value in neighbors_values if abs(G.nodes[node]['value'] - value) <= threshold]
            if diff_values:
                diff_weights = []
                for neighbor in G.neighbors(node):
                    if abs(G.nodes[node]['value'] - G.nodes[neighbor]['value']) <= threshold:
                        ##influence_degree = 1 / len(list(G.neighbors(neighbor)))
                        inf = G.nodes[node]['influence_degree']
                        diff_weights.append(inf)
                sum_weights = sum(diff_weights)
                diff_weights = [weight / sum_weights for weight in diff_weights]
                update_value = sum([diff_weights[i] * diff_values[i] for i in range(len(diff_values))])
                G.nodes[node]['value'] = update_value
    else:
        if neighbors_values:
            diff_values = [value for value in neighbors_values if abs(G.nodes[node]['value'] - value) <= threshold]
            if diff_values:
                diff_weights = []
                for neighbor in G.neighbors(node):
                    if abs(G.nodes[node]['value'] - G.nodes[neighbor]['value']) <= threshold:
                        ##influence_degree = 1 / len(list(G.neighbors(neighbor)))
                        inf = G.nodes[node]['influence_degree']
                        diff_weights.append(inf)
                sum_weights = sum(diff_weights)
                sum_weights = sum_weights + 1
                diff_weights = [weight / sum_weights for weight in diff_weights]
                update_value_neighbor = sum([diff_weights[i] * diff_values[i] for i in range(len(diff_values))])
                update_value_AI = (1 / sum_weights) * CHT
                G.nodes[node]['value'] = update_value_neighbor + update_value_AI
# def average_update(node, G):
#     neighbors_values = [G.nodes[neighbor]['value'] for neighbor in G.neighbors(node)]
#     G.nodes[node]['value'] = sum(neighbors_values) / len(neighbors_values)
def constant_update(node, G):
    G.nodes[node]['value'] = CHT

# Draw initial graph
fig, ax = plt.subplots(figsize=(8, 8))
pos = nx.spring_layout(G)
cmap = plt.cm.get_cmap('YlGnBu_r')
cmin = -1
cmax = 1
node_colors = [G.nodes[node]['value'] for node in G.nodes()]
node_sizes = [G.nodes[node]['influence_degree']*800 for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=cmap, vmin=cmin, vmax=cmax, ax=ax, node_size=node_sizes)
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, ax=ax)
sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=cmin, vmax=cmax))
sm.set_array([])
plt.colorbar(sm, ax=plt.gca())
plt.title(f"Iteration 0")
plt.axis('off')
plt.tight_layout()
fig.canvas.draw()
image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
plt.show
plt.savefig('initial network.png')


# Perform iterations
node_values = []
for i in range(iterations+1):
    values = [G.nodes[node]['value'] for node in G.nodes()]
    node_values.append(values)
    for node in HK_nodes:
        HK_update(node, G, threshold)
    for node in compre_nodes:
        compre_update(node, G, threshold)
    for node in constant_nodes:
        constant_update(node, G)

# Draw final graph
# fig, ax = plt.subplots(figsize=(8, 8))
# pos = nx.spring_layout(G)
# cmap = plt.cm.get_cmap('YlGnBu_r')
# cmin = -1
# cmax = 1
# node_colors = [G.nodes[node]['value'] for node in G.nodes()]
# node_sizes = [G.nodes[node]['influence_degree']*500 for node in G.nodes()]
# nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=cmap, vmin=cmin, vmax=cmax, ax=ax, node_size=node_sizes)
# nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, ax=ax)
# sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=cmin, vmax=cmax))
# sm.set_array([])
# plt.colorbar(sm, ax=plt.gca())
# plt.title(f"Iteration 100")
# plt.axis('off')
# plt.tight_layout()
# fig.canvas.draw()
# image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
# plt.show
# plt.savefig('final network.png')


# Plot trend graph
## plot line
plt.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(10,6))
HK_line, = plt.plot([],[], color='blue', alpha=0.5)
comp_line, = plt.plot([],[], color='green', alpha=0.5)
other_line, = plt.plot([],[], color='red', alpha=0.5)
lines = [HK_line, comp_line, other_line]
labels = ['NBN', 'NBNC', 'NBC']
colors = ['blue' if node in HK_nodes else 'green' if node in compre_nodes else 'red' for node in G.nodes()]
for node, color in zip(G.nodes(),colors):
    if color == 'blue':
        plt.plot(range(iterations+1), [values[node] for values in node_values], c = color, alpha=0.5)
    elif color == 'green':
        plt.plot(range(iterations+1), [values[node] for values in node_values], c = color, alpha=0.5)
    else:
        plt.plot(range(iterations+1), [values[node] for values in node_values], c = color, alpha=0.5)
plt.xlabel('Iteration')
plt.ylabel('Opinion')
plt.title('threshold:'+str(threshold)+', pro_NBN:'+str(pro_HK)+', pro_NBNC:'+str(pro_compre)+', pro_NBC:'+str(pro_constant)+' ')
plt.legend(lines, labels, prop={'weight':'bold'})
plt.show()
# deo = pd.DataFrame(node_values)
# deo.to_csv(r'C:\Users\liu\Desktop\dae2.csv')



# ## plot scatter
# plt.rcParams['font.family'] = 'Arial'
# plt.figure(figsize=(10,6))
# colors = ['blue' if node in HK_nodes else 'green' if node in compre_nodes else 'red' for node in G.nodes()]
# for node, color in zip(G.nodes(), colors):
#     values = [G.nodes[node]['value'] for i in range(iterations+1)]
#     plt.scatter(range(iterations+1), values, c=color, alpha=0.5)
# plt.xlabel('Iteration')
# plt.ylabel('Node Value')
# plt.title('Node Value Trends')
# plt.legend()
# plt.show()




