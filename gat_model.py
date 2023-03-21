from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
from torch_geometric.loader import DataLoader
import os
import networkx as nx
import matplotlib.pyplot as plt
import torch
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
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
# from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
import random
from skorch import NeuralNetClassifier
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader
curretn_path = os.getcwd()
path = f"{curretn_path}/chest_xray_graphs_adv_fgsm" # 75-78 % acc on adv
# path = f"{curretn_path}/chest_xray_graphs_simple_fgsm"

embed_dim = 128
X,Y = [],[]
def load_all_from_one_folder(path,type = None,train_test = 0):
    all_files = os.listdir(path)
    all_data = []
    k = 0
    # if type == 1:# and train_test == 1:
    #       all_files = all_files[0:1401] 
    
    for one_g in all_files:
        print(one_g)
        name = one_g.split(".")[0]
        G = nx.read_gpickle(f"{path}/{one_g}")  #map_location=torch.device('cpu')
        #G = nx.read_gpickle(torch.load(f"{path}/{one_g}",map_location=torch.device('cpu')))
        # print(G.nodes[0]['x'].shape)
        data = from_networkx(G)
        print(data)
        yy = [0]
        if type:
            data.y = [1]
            yy = [1]
        else:
            data.y = [0]
        k+= 1
        # print(data.x.shape)
        data.x = torch.Tensor([torch.flatten(val).tolist() for val in data.x])#nx.get_node_attributes(G,'image')
        data.name = name
        # data.x = data.x.type(torch.LongTensor)
        print(k,data)
        X.append([data])
        Y.append(yy[0])
        all_data.append(data)

    return all_data


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
    train_normal = load_all_from_one_folder(f"{path}/train/NORMAL",0,1)
    train_pneumonia = load_all_from_one_folder(f"{path}/train/PNEUMONIA",1)

    test_normal = load_all_from_one_folder(f"{path}/test/NORMAL")
    test_pneumonia = load_all_from_one_folder(f"{path}/test/PNEUMONIA",1)

    val_normal = load_all_from_one_folder(f"{path}/val/NORMAL")
    val_pneumonia = load_all_from_one_folder(f"{path}/val/PNEUMONIA",1)


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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False,drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True,drop_last=True)

    return train_loader, test_loader, train_dataset, test_dataset, val_loader, val_dataset



train_loader, test_loader, train_dataset, test_dataset, val_loader, val_dataset = dataloader()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data[0].edge_attr)
    print()

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        # self.conv1 = GCNConv(512, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # self.lin = Linear(hidden_channels, 2)
        self.conv1 = GCNConv(512, 512)
        self.conv2 = GCNConv(512, 256)
        self.conv3 = GCNConv(256,128)
        self.conv4 = GCNConv(128, 64)
        self.lin1 = Linear(64, 32)
        #self.lin2 = Linear(128,64)
        self.lin = Linear(32, 2)

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
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        #x = self.lin2(x)
        #x = x.relu()
        x = self.lin(x)
        
        return x

# model = GCN(hidden_channels=64)
# print(model)


# model = GCN(hidden_channels=64)



model = Sequential('x, edge_index, batch', [
    (Dropout(p=0.5), 'x -> x'),
    (GCNConv(1280, 512), 'x, edge_index -> x1'),
    ReLU(inplace=True),
    (GCNConv(512, 256), 'x1, edge_index -> x2'),
    ReLU(inplace=True),
    (GCNConv(256, 128), 'x2, edge_index -> x3'),
    ReLU(inplace=True),
    (GCNConv(128, 64), 'x3, edge_index -> x4'),
    ReLU(inplace=True),
    (GCNConv(64, 64), 'x4, edge_index -> x5'),
    ReLU(inplace=True),
    (lambda x4, x5: [x4, x5], 'x4, x5 -> xs'),
    (JumpingKnowledge("max", 64, num_layers=4), 'xs -> x'),
    # (JumpingKnowledge("lstm", 64, num_layers=2), 'xs -> x'),
    (global_mean_pool, 'x, batch -> x'),
    # Linear(2 * 64, 2),
    (Linear(64, 32), 'x -> x'),
    ReLU(inplace=True),
    (Linear(32, 2), 'x -> x'),
])


class GAT3(torch.nn.Module):
    def __init__(self):
        super(GAT3, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        
        self.conv1 = GATConv(512, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, 32, concat=False,
                             heads=8)
        self.conv3 = GATConv(32, 16, heads = 1, concat=True)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        self.lin1 = Linear(16, 2)

    def forward(self,x, edge_index,batch):
        
        # Dropout before the GAT layer is used to avoid overfitting
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin1(x)

        return x

# model = GAT3()


class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        
        self.conv1 = GATConv(512, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, 32, concat=False,
                             heads=self.out_head, dropout=0.6)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        self.lin1 = Linear(32, 2)

    def forward(self,x, edge_index,batch):
        
        # Dropout before the GAT layer is used to avoid overfitting
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin1(x)

        return x

class GAT2(torch.nn.Module):
    def __init__(self):
        super(GAT2, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1

        # self.conv1 = GATConv(1280, self.hid, heads=self.in_head, dropout=0.6)
        # self.conv2 = GATConv(self.hid*self.in_head, 32, concat=False,
        #                      heads=self.out_head, dropout=0.6)
        
        self.conv1 = GATConv(1280, 512, heads=self.in_head)
        self.conv2 = GATConv(4096, 256,heads=8)
        self.conv3 = GATConv(2048, 128, heads=4) #(4605x256 and 2048x512)
        self.conv4 = GATConv(512, 64, heads=1, concat=False)
        self.lin1 = Linear(64, 32)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        self.lin2 = Linear(32, 2)

    def forward(self,x, edge_index,batch):
        
        # Dropout before the GAT layer is used to avoid overfitting
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = self.conv4(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.6, training=self.training)
        # x = self.conv2(x, edge_index)
        x = self.lin1(x)
        x = F.elu(x)
        x = self.lin2(x)

        return x



model = GAT3()



optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        
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
label = []
predication = []
def test(loader, flag = 0):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         data.y = torch.Tensor(data.y)
        #  print("==="*10)
        #  print(data)
         pred = out.argmax(dim=1).view(-1,1)  # Use the class with highest probability.
        #  print(pred,"pred here",data.y)
         cf_matrix = confusion_matrix(data.y,pred)
         global cfm
         cfm = cf_matrix
         if flag:
            print(cfm)
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        #  label.append(data.y.cpu().numpy())
        #  predication.append(pred.cpu().numpy())
        #  print(label,"jhjgjhgjhgjhgjgj\n",predication)
         if flag:
            print(f"ROCAUC: {roc_auc_score(data.y.cpu().numpy(),pred.cpu().numpy(),average=None)}")
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 21):
    train()
    try:
        train_acc = test(train_loader)
        test_acc = test(test_loader,1)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        
    except Exception as e:
        print("error",e)
        pass
print(cfm)
print("number of paramteres for this model",sum(p.numel() for p in model.parameters()))
