# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:16:16 2020

@author: REZA
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



import networkx as nx

import imageio
from celluloid import Camera
from IPython.display import HTML
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

class GCNConv(nn.Module):
    def __init__(self, A, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.A_hat = A+torch.eye(A.size(0))
        self.D     = torch.diag(torch.sum(A,1))
        self.D     = self.D.inverse().sqrt()
        #self.D     = self.D.pow(-0.5)
        self.A_hat = torch.mm(torch.mm(self.D, self.A_hat), self.D)
        self.W     = nn.Parameter(torch.rand(in_channels,out_channels, requires_grad=True))
    
    def forward(self, X):
        out = torch.relu(torch.mm(torch.mm(self.A_hat, X), self.W))
        return out
class Net(torch.nn.Module):
    def __init__(self,A, nfeat, nhid, nout):
        super(Net, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid)
        self.conv2 = GCNConv(A,nhid, nout)
        
    def forward(self,X):
        H  = self.conv1(X)
        H2 = self.conv2(H)
        return H2

data=torch.Tensor([[0,1,1,0,1,0,0,1],
                   [1,0,1,0,0,1,1,0],
                   [1,1,0,1,1,0,1,0],
                   [0,0,1,0,1,0,1,1],
                   [1,0,1,1,0,1,1,0],
                   [0,1,0,0,1,0,0,1],
                   [0,1,1,1,1,0,0,0],
                   [1,0,0,1,0,1,0,0]
                   ])
                  
    
target=torch.tensor([0,-1,-1,-1,-1, -1, -1, 1])



graph = nx.Graph()


for i in range (data.shape[0]):
    graph.add_node(i)

for i in range (data.shape[0]):
    for j in range(data.shape[0]):
        if data[i][j]==1:
            graph.add_edge(i,j)  
            
nx.draw_networkx(graph)
print(nx.info(graph))





features=torch.Tensor([[10, 230, 147,0,0,0,0,0],
                       [5, 52, 200,0,0,0,0,0],
                       [3, 76, 260,0,0,0,0,0],
                       [21, 42, 100,0,0,0,0,0],
                       [0, 12, 16,0,0,0,0,0],
                       [1, 330, 280,0,0,0,0,0],
                       [33, 178, 15,0,0,0,0,0],
                       [11, 90, 96,0,0,0,0,0]
                       ])
    
X=torch.zeros(8,8)
    
T=Net(data,features.size(0), 10, 2)

#features=torch.eye(data.size(0))
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.SGD(T.parameters(), lr=0.01, momentum=0.9)
loss=criterion(T(features),target)
print(loss)

#%% Plot animation using celluloid
fig = plt.figure()
camera = Camera(fig)

for i in range(200):
    optimizer.zero_grad()
    loss=criterion(T(features), target)
    loss.backward()
    optimizer.step()
    l=(T(features));

    plt.scatter(l.detach().numpy()[:,0],
                l.detach().numpy()[:,1],c=[0, 0, 0, 0 ,1 ,1 ,0, 1])
    for i in range(l.shape[0]):
        text_plot = plt.text(l[i,0], l[i,1], str(i+1))

    camera.snap()

    if i%20==0:
        print("Cross Entropy Loss: =", loss.item())

animation = camera.animate(blit=False, interval=150)
animation.save('./reza.gif',writer='ffmpeg',  fps=60)
HTML(animation.to_html5_video())

'''

features=torch.Tensor([[10., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 1., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 1., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 1., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 1., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 1., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 10.]
                       ])
'''

















