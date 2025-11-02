import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import numpy as np
import h5py

# This reads the matlab data from the .mat file provided
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead).__init__()

        self.file_path = file_path
        self.data = h5py.File(self.file_path)

    def get_load(self):
        strain = np.array(self.data['load_apply']).transpose(1,0)
        return torch.tensor(strain, dtype=torch.float32)

    def get_result(self):
        stress = np.array(self.data['result']).transpose(1,0)
        return torch.tensor(stress, dtype=torch.float32)

# Define data normalizer
class DataNormalizer:
    def __init__(self, data):
        """
        Compute mean and standard deviation for normalization.

        """
        self.mean = data.mean(dim=0, keepdim=True)  # Compute mean along feature axis
        self.std = data.std(dim=0, keepdim=True)  # Compute standard deviation along feature axis

        # Prevent division by zero in case of constant features
        self.std[self.std == 0] = 1  

    def normalize(self, data):
        """
        Normalize the data using precomputed mean and std.

        """
        return (data - self.mean) / self.std

    def decode(self, normalized_data):
        """
        Decode the normalized data back to the original scale.

        """
        return (normalized_data * self.std) + self.mean


class ResNet1D(nn.Module):
    def __init__(self, input_dim, layer_dims, num_classes, activation):
        """
        A flexible ResNet for 1D input (tabular/sequential data).

        """
        super(ResNet1D, self).__init__()

        self.activation = activation()  # Store activation function

        # First layer to transform input
        self.input_layer = nn.Linear(input_dim, layer_dims[0])

        # Create residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(layer_dims[i], layer_dims[i+1], activation) for i in range(len(layer_dims) - 1)]
        )

        # Output layer
        self.output_layer = nn.Linear(layer_dims[-1], num_classes)

    def forward(self, x):
        x = self.activation(self.input_layer(x))  # Initial transformation
        x = self.residual_blocks(x)  # Pass through residual blocks
        x = self.output_layer(x)  # Output layer
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        """
        Residual block with flexible dimensions and activation.
        
        """
        super(ResidualBlock, self).__init__()
        self.activation = activation()  # Store activation function

        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

        # If input and output dimensions do not match, use a linear layer to adjust
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)  # Transform input if necessary
        out = self.activation(self.fc1(x))
        out = self.fc2(out)  # No activation on last layer

        out += identity  # Skip connection (adds input to output)
        return self.activation(out)  # Apply activation after residual sum


######################### Data processing #############################
# Read data from .mat file
path = 'Eiffel_data.mat' 
data_reader = MatRead(path)
load = data_reader.get_load()
result = data_reader.get_result()

# Split data into train and test
total_samples = load.shape[0]
ntrain = 900 
ntest = 100  

# Generate shuffled indices
indices = torch.randperm(total_samples)

# Split data
train_indices = indices[:ntrain]  
test_indices = indices[ntrain:ntrain + ntest]  

# Apply shuffling
train_load = load[train_indices]
train_result = result[train_indices]
test_load = load[test_indices]
test_result = result[test_indices]

# Normalize your data
load_normalizer   = DataNormalizer(train_load)
train_load_encode = load_normalizer.normalize(train_load)
test_load_encode  = load_normalizer.normalize(test_load)

ndim = load.shape[1]  # Number of components

# Create data loader
batch_size = 100
train_set = Data.TensorDataset(train_load_encode, train_result)
train_loader = Data.DataLoader(train_set, batch_size, shuffle=True)
print(len(train_loader))

test_set = Data.TensorDataset(test_load_encode, test_result)
test_loader = Data.DataLoader(test_set, batch_size, shuffle=False)
print(len(test_loader))

############################# Define and train network #############################
# Create Nueral network, define loss function and optimizer

# Device configuration (Optional)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 20
output_size = 1

num_epochs = 30
learning_rate = 0.01
ResNet_arch = [64, 128, 64]
non_linearity = nn.Sigmoid 
net = ResNet1D(input_size, ResNet_arch, output_size, non_linearity)
print(net)

n_params = sum(p.numel() for p in net.parameters() if p.requires_grad) #Calculate the number of training parameters
print('Number of parameters: %d' % n_params)

loss_func = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) 

# Train network
epochs = num_epochs  
print("Start training for {} epochs...".format(epochs))

loss_train_list = []
accuracy_list = []

for epoch in range(epochs):
    net.train(True)
    trainloss = 0


    for i, data in enumerate(train_loader):
        input, target = data
        optimizer.zero_grad()
        
        output = net(input)
        
        loss = loss_func(output, target) # Calculate loss

        loss.backward()                    
        optimizer.step()                   

        trainloss += loss.item()         

    net.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = len(test_loader.dataset)

        for i, data in enumerate(test_loader):
            input, target = data
            output = net(input)
            
            predicted = (output > 0.5).float()
            n_correct += (predicted == target).sum().item()

        acc = n_correct / n_samples

    scheduler.step()

    print ("end epoch", epoch, "Train loss", loss.item(), "Accuracy", acc)

    loss_train_list.append(trainloss/len(train_loader))
    accuracy_list.append(acc)


print("Train loss:{}".format(trainloss/len(train_loader)))
print("Accuracy:{}".format(accuracy_list))

############################# Plot your result below using Matplotlib #############################
plt.figure(1)
epochs = range(1, len(loss_train_list) + 1)

# Plot loss over epochs
plt.plot(epochs, loss_train_list, marker='o', linestyle='-')
plt.title('Train and Test Losses')

plt.figure(2)

# Plot input vs. output
plt.plot(epochs, accuracy_list, marker='o', linestyle='-')
plt.title('Accuracy')
plt.grid(True)
plt.show()

