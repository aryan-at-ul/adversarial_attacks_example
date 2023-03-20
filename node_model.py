from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
from torch_geometric.loader import DataLoader
import os
import networkx as nx
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
# from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
import random
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
import math
from torch_geometric.nn import SAGEConv

curretn_path = os.getcwd()
path = f"{curretn_path}/chest_xray_graphs"
# path = f"{curretn_path}/chest_xray_graphs_nodes_disf_nodeclass_resnet18_date_march7th"
embed_dim = 128

class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_len=100):

        super(PositionalEncoding, self).__init__()


        pe = torch.zeros(max_len, dim)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-(math.log(10000.0) / dim)))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):

        x = x + self.pe[:, :x.size(1), :]
        return x


def add_positional_encoding(data_list, max_len = 100):

    print("Adding positional encoding",max_len, len(data_list))
    k = 0 
    for data in data_list:
        node_features = data.x
        node_features = node_features.unsqueeze(0)
        pos_enc = PositionalEncoding(node_features.size(-1), max_len)
        pos_enc_features = pos_enc(node_features)
        pos_enc_features = pos_enc_features.squeeze(0)
        data.x = torch.cat((data.x,pos_enc_features),dim=0)
        if k == 0:
            print(node_features.shape,pos_enc_features.shape)
            print(data.x.shape)
            k+=1

    return data_list


def remove_edges(G):
    p = 0.2# Iterate over the edges and drop them with probability p
    edges_to_remove = []
    for u, v in G.edges():
        if random.random() < p:
            edges_to_remove.append((u, v))
    for u, v in edges_to_remove:
        G.remove_edge(u, v)
    return G


global_graph_disc = {}
# kg = 0 
def load_all_from_one_folder(path,kg,type = 0,train_test = 0):
    all_files = os.listdir(path)
    all_data = []
    k = 0
    # if type == 1 and train_test == 1:
    #     all_files = np.random.choice(all_files, size=2000, replace=False)#all_files[0:1301] 

    if type == 0 and train_test == 1:
        more_files = np.random.choice(all_files, size=2000, replace=True)
        all_files = np.concatenate((all_files,more_files),axis=0)


    for one_g in all_files:
        # print(one_g)
        name = one_g.split(".")[0]
        pp = 0.5
        try:
            G = nx.read_gpickle(f"{path}/{one_g}")  #map_location=torch.device('cpu')
            # print(G)
            if train_test == 1 and random.random() < pp:
                G = remove_edges(G)
            #G = nx.read_gpickle(torch.load(f"{path}/{one_g}",map_location=torch.device('cpu')))
            # print(G,"this is G",G.nodes.data())['label']
            data = from_networkx(G)
            # print(data)
        except Exception as e:
            print("error",e)
            # sys.exit()
            continue

        if type:
            data.y = [1]
        else:
            data.y = [0]
        k+= 1
        kg += 1
        # print(data.x.shape)
        # print(data.label,"this is label")

        data.x = torch.Tensor([torch.flatten(val).tolist() for val in data.x])#nx.get_node_attributes(G,'image')
        data.labels = torch.Tensor([torch.flatten(val).tolist() for val in data.label])
        print(name,len(data.label))
        names = [name]
        global_graph_disc[float(kg)] = name
        data.name = torch.Tensor([kg]*len(data.label))
        data.name = torch.Tensor([torch.flatten(val).tolist() for val in data.name])
        # print(data.name,"names in data ",global_graph_disc)
        # data.name = torch.Tensor([torch.flatten(val).tolist() for val in [torch.Tensor(k)]])
        # names = torch.Tensor(k)
        # print("names",names)
        # names = names.repeat(len(data.x),1)
        # data.name = names
        # data.x = data.x.type(torch.LongTensor)
        print("final changes",k,data)
        all_data.append(data)
    return all_data,kg


def permute_array(array):
    permuted_array = []
    for i in range(len(array)):
        permuted_array.append(array[i])
    return permuted_array




def dataloader():
    """
    load train and test data
    """
    print("loading data")
    kg = 0
    train_normal,kg = load_all_from_one_folder(f"{path}/train/NORMAL",kg,0,1)
    train_pneumonia,kg  = load_all_from_one_folder(f"{path}/train/PNEUMONIA",kg,1,1)

    test_normal,kg  = load_all_from_one_folder(f"{path}/test/NORMAL",kg)
    test_pneumonia,kg  = load_all_from_one_folder(f"{path}/test/PNEUMONIA",kg,1)

    val_normal,kg  = load_all_from_one_folder(f"{path}/val/NORMAL",kg)
    val_pneumonia,kg  = load_all_from_one_folder(f"{path}/val/PNEUMONIA",kg,1)


    train_data_arr = train_normal + train_pneumonia
    test_data_arr = test_normal + test_pneumonia
    val_data_arr = val_normal + val_pneumonia
    # all_data = permute_array(all_data)
    random.shuffle(train_data_arr)
    random.shuffle(test_data_arr)
    random.shuffle(val_data_arr)
    
    #if True:
    #    transform = T.GDC(
    #    self_loop_weight=1,
    #    normalization_in='sym',
    #    normalization_out='col',
    #    diffusion_kwargs=dict(method='ppr', alpha=0.05),
    #    sparsification_kwargs=dict(method='topk', k=128, dim=0),
    #    exact=True,
    #    )
    #    data = transform(val_data_arr[0])


    train_dataset = train_data_arr#all_data[:int(len(all_data)*0.8)]
    val_dataset = val_data_arr#all_data[int(len(all_data)*0.8):int(len(all_data)*0.8) + 100]
    test_dataset = test_data_arr#all_data[int(len(all_data)*0.8):]
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False,drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False,drop_last=True)
    print(len(train_dataset),"kjdfkhdkfkdvnkdfnvkjndfivdnkjfdnvk")
    # train_dataset = add_positional_encoding(train_dataset.copy(),51)
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)

    # test_dataset = add_positional_encoding(test_dataset.copy(),51)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False,drop_last=True)



    return train_loader, test_loader, train_dataset, test_dataset, val_loader, val_dataset



train_loader, test_loader, train_dataset, test_dataset, val_loader, val_dataset = dataloader()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


   

#val_dataset = add_positional_encoding(val_loader.dataset,11)
#val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False,drop_last=True)

import sys





for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels = 1280):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        # self.conv1 = GCNConv(512, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # self.lin = Linear(hidden_channels, 2)
        self.conv1 = GCNConv(1280, 256)
        self.conv2 = GCNConv(256, 128)
        self.conv3 = GCNConv(128,64)
        self.conv4 = GCNConv(64, 32)
        self.lin1 = Linear(32, 128)
        self.lin2 = Linear(128,64)
        self.lin = Linear(64, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        #x = JumpingKnowledge(mode = 'cat')(x)
        # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.lin(x)
        
        return F.log_softmax(x, dim=1)
# model = GCN(hidden_channels=64)


class GCN(torch.nn.Module):
    def __init__(self, num_features = 2208, hidden_channels = 512, num_classes = 2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = 0.3,training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)               
        x = F.dropout(x, p = 0.3,training=self.training)
        x = self.conv3(x, edge_index)           
        x = F.relu(x)       
        x = F.dropout(x, p = 0.3,training=self.training)
        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=8, dropout=0.6):
        super(GAT, self).__init__()

        self.num_layers = num_layers

        # First GAT layer
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)

        # Intermediate GAT layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))

        # Last GAT layer
        self.conv_last = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)

    def forward(self, x, edge_index):
        # First GAT layer
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        
        # Intermediate GAT layers
        for i in range(self.num_layers - 2):
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.relu(self.convs[i](x, edge_index))

        # Last GAT layer
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv_last(x, edge_index)

        return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)

class GraphSAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GraphSAGENet, self).__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.conv_out = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.conv_out(x, edge_index)
        return F.log_softmax(x, dim=-1)


model = GCN()
model  = GAT(2208, 512, 2, 4)
#model = GraphSAGE(512,256,2)
model = GraphSAGENet(2208,512,2,4)

print(model)


model = GCN(hidden_channels=64)#.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.001)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        # data = data.to(device)
        #x, y, graph_id = batch['x'], batch['y'], batch['graph_id']
        graph_id = data.name
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        data.y = torch.Tensor(data.y)
        data.y = torch.Tensor(torch.flatten(data.y))
        data.y = data.y.type(torch.LongTensor)
        # print(data.y,"kjhkjdhsfkjhsdkjfhksjdhfkjsdhkjfhsdkjhfjs")
        # print(out,"dsjflkdsjlfkjsdlkfjlkdsjflksdjlfkjsdlkjlkj")
        loss = criterion(out, data.y)
        #print(loss.item())
        #loss = nn.BCELoss(out,data.y)
        #loss = F.nll_loss(out, data.y)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.





cfm = None

def test(loader, epoch,flag = 0):
     model.eval()
     prev_best = 0.84
     correct = 0
     accuracy = 0
     k = 0
     acc = 0
     graph_id_to_merege_all = []
     datay_id_to_merege_all = []
     pred_id_to_merege_all = []
     for data in loader:  # Iterate in batches over the training/test dataset.
        #  acc = 0
         #data = data.to(device)
         graph_idx = data.name
         out = model(data.x, data.edge_index, data.batch)  
         data.y = torch.Tensor(data.y)
   
         pred = out.argmax(dim=1).view(-1,1)  
         correct = (pred == data.y).sum().item() 

         graph_ids = list(set(torch.flatten(graph_idx).tolist()))
         graph_ids = [int(i) for i in graph_ids]

         graph_id_to_merege = torch.flatten(graph_idx).tolist()
         datay_id_to_merege = torch.flatten(data.y).tolist()
         pred_id_to_merege = torch.flatten(pred).tolist()

         graph_id_to_merege_all += [global_graph_disc[i] for i in graph_id_to_merege]
         datay_id_to_merege_all += [int(i) for i in datay_id_to_merege]
         pred_id_to_merege_all += [int(i) for i in pred_id_to_merege]
        #  print(global_graph_disc[1.0],len(global_graph_disc.keys()))
        #  print(graph_id_to_merege.shape,"graph id to merege",data.y.shape,"data y shape",pred.shape,"pred shape")

         accuracy = correct/data.y.shape[0]
         acc += accuracy
         k += 1
         if flag:
            # print(graph_id_to_merege_all)
            print("validation accuracy",accuracy,"total",data.y.shape[0],"correct",correct)
    

     if epoch == 50 and flag:
     
        data = {'filename': graph_id_to_merege_all, 'truevalue': datay_id_to_merege_all, 'prediction': pred_id_to_merege_all}

        df = pd.DataFrame(data)

        df.to_csv('testnodepreds.csv', index=False)

     if flag:
        return accuracy
     else:
        # print("acc",acc,"k",k)
        return acc / k    
        
    #  return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1,51):
    train()
    # try:
    train_acc = test(train_loader,epoch)
    test_acc = test(test_loader,epoch,1)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    # except Exception as e:
    # print("error",e)
    # pass
# print(cfm)
print("number of paramteres for this model",sum(p.numel() for p in model.parameters()))
