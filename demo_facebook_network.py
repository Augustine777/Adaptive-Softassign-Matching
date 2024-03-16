import networkx
import numpy as np

nosiy_graph = networkx.read_leda("C:/Users/25493/Desktop/networks/networks/synthetic_nets_known_node_mapping/low_confidence/0Krogan_2007_high+25e.gw")
sorce_graph = networkx.read_leda("C:/Users/25493/Desktop/networks/networks/synthetic_nets_known_node_mapping/0Krogan_2007_high.gw")
adj_noisy = networkx.to_numpy_matrix(nosiy_graph)
adj_sorce = networkx.to_numpy_matrix(sorce_graph)

M, runtime = graphmatch_ASM(adj_noisy, adj_sorce,adaptive_alpha=1,niter_max=40,tol=0.1)

print('Running time is '+ str(runtime)+ 's')
Accuracy = sum(np.diag(M))/4039 # The correct matching matrix is Identity matrix and number of nodes is 4039
print('Accuracy is '+ str(Accuracy)) 
