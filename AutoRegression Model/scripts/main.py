from classes_functions import *
from dnn import *
import scipy.io
from torch.utils.data import DataLoader
import torch.optim as optim

mat_contents = scipy.io.loadmat("../data/data_NN")

X_old = torch.zeros((3,1000),dtype=torch.float)
Y_old = torch.zeros((3,1000),dtype=torch.float)

#load data
for i in range (3):
    X_old[i] = torch.from_numpy(mat_contents['x_train'][i][:-1])
    Y_old[i] = torch.from_numpy(mat_contents['x_train'][i][1:])
    
    
X = torch.transpose(X_old.detach().clone(), 0,1)
Y = torch.transpose(Y_old.detach().clone(), 0,1)

minimum = []
minimum.append(torch.min(X_old[0]).item())
minimum.append(torch.min(X_old[1]).item())
minimum.append(torch.min(X_old[2]).item())

maximum = []
maximum.append(torch.max(X_old[0]).item())
maximum.append(torch.max(X_old[1]).item())
maximum.append(torch.max(X_old[2]).item())

  
#normalize the input data
normalize_input(X,maximum,minimum)
dyn_dataset = DynamicData(X,Y)


EPOCHS = 1000
BATCH_SZ = 128
net = MyPredictionNet([], 3, 7)  

print(f"The network architecture is: \n {net}")

# define the loss function in which case is CrossEntropy Loss
criterion = nn.MSELoss() 

ETA = 1e-2 
# define the optimizer which will be used to update the network parameters
optimizer = optim.Adam(net.parameters(), lr=ETA, weight_decay=1e-7) # feed the optimizer with the netowrk parameters

# here we use th sklearn built-in split function
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dyn_dataset,(800, 100, 100))

train_dl = DataLoader(train_dataset, batch_size=BATCH_SZ)
val_dl = DataLoader(val_dataset, batch_size=BATCH_SZ)
test_dl = DataLoader(test_dataset, batch_size=BATCH_SZ)

best_model = train(net, EPOCHS, train_dl, val_dl, optimizer, criterion, maximum, minimum)
