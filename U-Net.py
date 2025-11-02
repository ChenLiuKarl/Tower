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
        self.mean = data.mean(dim=0, keepdim=True)  
        self.std = data.std(dim=0, keepdim=True)  

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

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet1D, self).__init__()

        # Encoder (Downsampling Path)
        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(DoubleConv1D(in_channels, feature))
            in_channels = feature

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv1D(features[-1], features[-1] * 2)

        # Decoder (Upsampling Path)
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv1D(feature * 2, feature))

        # Final Convolution Layer
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        # Encoder
        skip_connections = []
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]  # Reverse for skip connections
        for i in range(len(self.decoder)):
            x = self.upconvs[i](x)  # Upsample
            skip_connection = skip_connections[i]

            # Ensure correct size match (due to rounding in pooling)
            if x.shape[-1] != skip_connection.shape[-1]:
                x = F.interpolate(x, size=skip_connection.shape[-1])

            x = torch.cat((skip_connection, x), dim=1)  # Concatenate skip connection
            x = self.decoder[i](x)

        # Final activation for boolean classification 
        x = self.final_conv(x) # Raw logits 
        x = torch.sigmoid(x) # Convert to probability (0 to 1) 
        return x


class DoubleConv1D(nn.Module):
    """ Two sequential 1D convolutions followed by ReLU """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

######################### Data processing #############################
# Read data from .mat file
path = 'Eiffel_data.mat'
data_reader = MatRead(path)
load = data_reader.get_load()
result = data_reader.get_result()

load_u = torch.zeros(load.shape[0],1,load.shape[1])
result_u = torch.zeros(load.shape[0],1,load.shape[1])

load_u[:,0,:] = load

# Split data into train and test
total_samples = load.shape[0]
ntrain = 900 
ntest = 100  

# Generate shuffled indices
indices = torch.randperm(total_samples)

# Split data
train_indices = indices[:ntrain]  
test_indices = indices[ntrain:ntrain + ntest] 

train_load = load_u[train_indices]
train_result = result[train_indices]
test_load = load_u[test_indices]
test_result = result[test_indices]

# Normalize your data
load_normalizer   = DataNormalizer(train_load)
train_load_encode = load_normalizer.normalize(train_load)
test_load_encode  = load_normalizer.normalize(test_load)

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

num_epochs = 30
learning_rate = 0.0015
net = UNet1D(in_channels=1, out_channels=1, features=[64, 128, 256])
print(net)

n_params = sum(p.numel() for p in net.parameters() if p.requires_grad) #Calculate the number of training parameters
print('Number of parameters: %d' % n_params)

loss_func = nn.BCELoss()
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
        output, indices = torch.max(output, dim=2)
        loss = loss_func(output, target) 

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
            output, indices = torch.max(output, dim=2)

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