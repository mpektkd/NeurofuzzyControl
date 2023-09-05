import copy
import torch 
from torch.utils.data import Dataset

#min-max normalization of the input at the 3 dimensions
def normalize_input(c, maximum, minimum, vector = 'false'): 
    if vector == 'true':
        c[0] = ((c[0] - minimum[0])/(maximum[0] - minimum[0]))
        c[1] = ((c[1] - minimum[1])/(maximum[1] - minimum[1]))
        c[2] = ((c[2] - minimum[2])/(maximum[2] - minimum[2]))
    
    else:
        a = torch.transpose(c,0,1)
        a[0] = ((a[0] - minimum[0])/(maximum[0] - minimum[0]))
        a[1] = ((a[1] - minimum[1])/(maximum[1] - minimum[1]))
        a[2] = ((a[2] - minimum[2])/(maximum[2] - minimum[2]))
        c = torch.transpose(a,0,1)

#transform data to the initial range for the correct comparisons with the output
def unormalize_output(c, maximum, minimum, vector = 'false'):
    if vector == 'true':
        c[0] = ((c[0]*(maximum[0] - minimum[0])) + minimum[0])
        c[1] = ((c[1]*(maximum[1] - minimum[1])) + minimum[1])
        c[2] = ((c[2]*(maximum[2] - minimum[2])) + minimum[2])
    
    else:
        a = torch.transpose(c,0,1)
    
        a[0] = ((a[0]*(maximum[0] - minimum[0])) + minimum[0])
        a[1] = ((a[1]*(maximum[1] - minimum[1])) + minimum[1])
        a[2] = ((a[2]*(maximum[2] - minimum[2])) + minimum[2])
    
        c = torch.transpose(a,0,1)

#Class for data passing to the DNN
class DynamicData(Dataset):
    def __init__(self, X,Y):
        self.data = list(zip(X,Y))   
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]
    
def train(net, epochs, train_dl, val_dl, optimizer, criterion, max, min):

    error = []
    for epoch in range(epochs): # loop through dataset
        net.train() # gradients "on"  
        for i, data in enumerate(train_dl): # loop through batches
            X_batch, Y_batch = data # get the features and labels
            optimizer.zero_grad() # ALWAYS USE THIS!! 
            out = net(X_batch) # forward pass
            unormalize_output(out, max,min) #unormalize the output
            loss = criterion(out, Y_batch) # compute per batch loss 
            loss.backward() # compurte gradients based on the loss function
            optimizer.step() # update weights

                
        net.eval() # turns off batchnorm/dropout
        running_average_loss = 0
        with torch.no_grad(): # no gradients required!! eval mode, speeds up computation
            for i, data in enumerate(val_dl):
                X_batch, Y_batch = data # test data and labels
                out = net(X_batch) # get net's predictions
                unormalize_output(out, max,min)#unormalize the output
                loss = criterion(out, Y_batch) # compute per batch loss 
                running_average_loss += loss.detach().item()
        if epoch == 0:
            error.append(running_average_loss)
            error.append(copy.deepcopy(net))
            error.append(epoch)
        if error[0] > running_average_loss:
            error[0] = running_average_loss
            error[1] = copy.deepcopy(net)
            error[2] = epoch
        
    best_model = copy.deepcopy(error[1])
    print("Less error: {} \t Epoch: {}".format(error[0], error[2]))

    return best_model