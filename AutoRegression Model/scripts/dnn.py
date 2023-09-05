import torch.nn as nn

#layer class with liearity and ReLU
class LinearWActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearWActivation,self).__init__()
        self.f = nn.Linear(in_features,out_features)  
        self.a = nn.ReLU()
    def forward(self,x):
        return self.a(self.f(x))
        
#Model that has to be trained
class MyPredictionNet(nn.Module):
    
    def __init__(self, layers, n_features, o_features):
        super(MyPredictionNet, self).__init__()
        layers_in = [n_features] + layers
        layers_out = layers + [o_features]
        self.f = nn.Sequential(
            *[LinearWActivation(in_feats, out_feats) 
             for in_feats, out_feats in zip(layers_in,layers_out)])
        self.clf = nn.Linear(o_features, n_features)
        
    def forward(self, x):
        return self.clf(self.f(x))
    