import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Yelp

import os

import torch_geometric.utils as utils

import matplotlib.pyplot as plt

from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"
print('Count of using GPUs:', torch.cuda.device_count())

Dataset_List = {}
# Dataset_List["Reddit"]=Reddit(root='/root/GNN_Datasets/Reddit')
# Dataset_List["Yelp"]=Yelp(root='/root/GNN_Datasets/Yelp')
# Dataset_List["Computers"]=Amazon(root='/root/GNN_Datasets/Amazon', name='Computers')
# Dataset_List["cs"]=dataset_cs = Coauthor(root='/root/GNN_Datasets/cs', name='cs')
Dataset_List["Cora"]=dataset_Cora = Planetoid(root='/root/GNN_Datasets/Cora', name='Cora')
# Dataset_List["Physics"]=dataset_Physics = Coauthor(root='/root/GNN_Datasets/cs', name='Physics')
# Dataset_List["Amazon_Photo"]=dataset_Amazon = Amazon(root='/root/GNN_Datasets/Amazon', name='Photo')

########################################################################################################
# Plot Adjacency Matrix
########################################################################################################
for _, Dataset in Dataset_List.items():

    edge_index = Dataset[0].edge_index
    print("Degree of Graph :", utils.degree(edge_index[0]))
    print("Number of Node : ", Dataset[0].num_nodes)
    print("Dataset : ", _)
    Adj = utils.add_self_loops(edge_index)
    Adj = Adj[0]
    print("Size of Adj :", Adj.size())
    plt.scatter(Adj[0:, :], Adj[1, :], s=0.01)
    plt.show()
    print("-"*200)

########################################################################################################
# Print Node Feature Sparsity
########################################################################################################
for Name, Dataset in Dataset_List.items():
    print('-'*100)
    print("Dataset : ", Name)
    print("Feature Vector Size : ", Dataset[0].x.size(dim=1))
    x = Dataset[0].x
    Non_Zero = torch.count_nonzero(x)
    x_size = x.size(dim=0) * x.size(dim=1)
    print("Sparsity of Feature Matrix : ", 1-int(Non_Zero)/int(x_size))
    print("x : ", x.dtype)


########################################################################################################
# GCN Model Definition(Using Cora Dataset(Smallest Dataset)
########################################################################################################

dataset = Dataset_List["Cora"]

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        H0 = data.x
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        H1 = x
        x = self.conv2(x, edge_index)
        H2 = x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        return F.log_softmax(x, dim=1), H0, H1, H2          # H0, H1, H2 are just for debugging



########################################################################################################
# Training
########################################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out, _, __, ___ = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
   



########################################################################################################
# Inference
########################################################################################################

model.eval()
pred, H0, H1, H2 = model(data)
pred = pred.argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
