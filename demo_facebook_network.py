import networkx
import numpy as np
import asm as *


G_face = networkx.read_gpickle("networks/facebook/G_face.gpickle")
G_face_noise_5 = networkx.read_gpickle("networks/facebook/G_face_noise_5.gpickle")
G_face_noise_15 = networkx.read_gpickle("networks/facebook/G_face_noise_15.gpickle")
G_face_noise_25 = networkx.read_gpickle("networks/facebook/G_face_noise_25.gpickle")

adj = networkx.to_numpy_matrix(G_face)
adj_5 = networkx.to_numpy_matrix(G_face_noise_5)
adj_15 = networkx.to_numpy_matrix(G_face_noise_15)
adj_25 = networkx.to_numpy_matrix(G_face_noise_25)

adj = networkx.to_numpy_matrix(G_face)
adj_5 = networkx.to_numpy_matrix(G_face_noise_5)
adj_15 = networkx.to_numpy_matrix(G_face_noise_15)
adj_25 = networkx.to_numpy_matrix(G_face_noise_25)

M, runtime = Matching.graphmatch_ASM(adj, adj_25,adaptive_alpha=1)
print('Running time is '+ str(runtime)+ 's')
Accuracy = sum(np.diag(M))/4039 # The correct matching matrix is the Identity matrix, and the number of nodes is 4039
print('Accuracy is '+ str(Accuracy))
