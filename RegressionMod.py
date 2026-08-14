##### imports #####
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler, ConcatDataset, Subset
from torch.profiler import profile, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision.transforms as T

##### globals #####
path = r'/Users/trentstarkey/Desktop'
image_dir = path + '/RegressionData_30kV_0.09nA'
val_dir = path + '/RegressionData_30kV_0.09nA_val'
csv_file = path + '/RegressionData_30kV_0.09nA/labels.csv'
csv_file_val = path + '/RegressionData_30kV_0.09nA_val/labels.csv'
test_dir = path + '/RegressionData_30kV_0.09nA_test'
csv_file_test = path + '/RegressionData_30kV_0.09nA_test/labels.csv'

batch = 12
learning_rate = 1e-2
mod_name = '2.0'
epochs = 30

USE_PROFILER = False

##### define functions #####
class Data(Dataset):
    def __init__(self, image_dir, csv_file, transform = None):
        self.image_dir = image_dir
        self.transform = transform

        self.data = pd.read_csv(csv_file)
        self.data['Image'] = self.data['Image'].astype(str)
        self.data['Defocus'] = pd.to_numeric(self.data['Defocus'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row['Image'])
        image = Image.open(image_path + '.jpeg').convert('L')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row['Defocus'], dtype=torch.float32)

        return image, label

def create_datasets():
    transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    generator = torch.Generator().manual_seed(1)
    
    #----- create datasets -----#
    dataset = Data(image_dir = image_dir, csv_file = csv_file, transform = transform)
    dataset1 = Data(image_dir = val_dir, csv_file = csv_file_val, transform = transform)
    indices = torch.randperm(len(dataset1), generator = generator).tolist()
    
    #----- create training and val split -----#
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset0, val_dataset0 = random_split(dataset, [train_size, val_size], generator = generator)
    
    #----- create secondary training and val split -----#
    train_fraction1 = 0.1
    val_fraction1 = 0.4
    
    train_size1 = int(train_fraction1 * len(dataset1))
    val_size1 = int(val_fraction1 * len(dataset1))

    train_indices1 = indices[:train_size1]
    val_indices1 = indices[train_size1:train_size1 + val_size1]
    train_dataset1 = Subset(dataset1, train_indices1)
    val_dataset1 = Subset(dataset1, val_indices1)
    
    val_dataset = ConcatDataset([val_dataset0, val_dataset1])
    train_dataset = ConcatDataset([train_dataset0, train_dataset1])

    #------ create random samplers to ensure even data representation ------#
    train_labels0 = dataset.data.iloc[train_dataset0.indices]['Defocus']
    train_counts0 = train_labels0.value_counts()
    train_weights0 = train_labels0.map(lambda x: 1.0 / np.sqrt(train_counts0[x])).values

    train_labels1 = dataset1.data.iloc[train_dataset1.indices]['Defocus']
    train_counts1 = train_labels1.value_counts()
    train_weights1 = train_labels1.map(lambda x: 1.0 / np.sqrt(train_counts1[x])).values
    
    train_weights = np.concatenate([train_weights0, train_weights1])
    sampler_train = WeightedRandomSampler(weights = torch.DoubleTensor(train_weights), num_samples=len(train_dataset), replacement = True)

    val_labels0 = dataset.data.iloc[val_dataset0.indices]['Defocus']
    val_counts0 = val_labels0.value_counts()
    val_weights0 = val_labels0.map(lambda x: 1.0 / np.sqrt(val_counts0[x])).values

    val_labels1 = dataset1.data.iloc[val_dataset1.indices]['Defocus']
    val_counts1 = val_labels1.value_counts()
    val_weights1 = val_labels1.map(lambda x: 1.0 / np.sqrt(val_counts1[x])).values
    
    val_weights = np.concatenate([val_weights0 , val_weights1])
    sampler_val = WeightedRandomSampler(weights = torch.DoubleTensor(val_weights), num_samples=len(val_dataset), replacement = False)

    #----- create final dataloaders -----#
    train_loader = DataLoader(train_dataset, batch_size = batch, sampler = sampler_train)
    val_loader = DataLoader(val_dataset, batch_size = batch, sampler = sampler_val, shuffle = False)

    return train_loader, val_loader

class ExpReLU(nn.Module):
       def forward(self, x):
            x = torch.maximum(x * torch.exp(torch.clamp(x, max = 10)), torch.tensor(0., device = x.device))
            return x 

class FFT(nn.Module):
    def forward(self, x):
        x = x[:, :, 3:-3, 3:-3]
        background = T.GaussianBlur((1, 1), sigma = (28, 28))
        background_x = background(x)
        x = (x - background_x)
        x = x / (torch.amax(x, dim=(-2, -1), keepdim=True) + 1e-8)

        x = torch.fft.fft2(x)
        x = torch.fft.fftshift(x, dim=(-2, -1))

        return torch.log1p(torch.abs(x))

class IFFTShift(nn.Module):
    def forward(self, x):
        return torch.fft.ifftshift(x)

class Patches(nn.Module):
    def __init__(self, patch_size = 4, resize_x = 200, resize_y = 200):
        super().__init__()
        self.patch_size = patch_size
        self.resize = nn.Upsample(size = (resize_x,resize_y), mode = 'bilinear', align_corners = False)

    def forward(self, x):
        x = self.resize(x)
        x = F.unfold(x, kernel_size = self.patch_size, stride = self.patch_size)

        B, C, L = x.shape
        h = 200 // self.patch_size
        w = 200 // self.patch_size
        x = x.view(B, C, h, w)

        return x

class DefocusRegressionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.expRELU = ExpReLU()

        self.pad1 = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(1,16, kernel_size = 3)
        self.bn1 = nn.BatchNorm2d(16)

        self.pad2 = nn.ZeroPad2d(1)
        self.conv2 = nn.Conv2d(16,32, kernel_size = 3)
        self.bn2 = nn.BatchNorm2d(32)

        self.res1 = nn.Conv2d(16,32, kernel_size = 3, padding = 1)

        self.dropout = nn.Dropout(0.05)

        self.fft = FFT()
        self.fft_conv1= nn.Conv2d(32, 128, kernel_size = 1)
        self.fft_conv2= nn.Conv2d(128, 256, kernel_size = 1)
        self.bn3 = nn.BatchNorm2d(256)

        self.fft_res = nn.Conv2d(32, 256, kernel_size = 1)

        self.ifft_shift = IFFTShift()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lin1 = nn.Linear(256, 2048)
        self.lin2 = nn.Linear(2048, 1024)
        self.output = nn.Linear(1024,1)

    def forward(self,x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.expRELU(x)
        x = F.max_pool2d(x,2)
        x = self.bn1(x)

        activation = x

        x = self.pad2(x)
        x = self.conv2(x)
        x = self.expRELU(x)
        x = F.max_pool2d(x,2)
        x = self.bn2(x)

        res = self.res1(activation)
        res = F.interpolate(res, size = x.shape[2:], mode = 'bilinear', align_corners = False)

        x = x + res
        activation = x

        x = self.dropout(x)

        x = self.fft(x)
        x = self.fft_conv1(x)
        x = self.expRELU(x)
        x = self.fft_conv2(x)
        x = self.expRELU(x)
        x = F.max_pool2d(x,2)
        x = self.bn3(x)

        res = self.fft_res(activation)
        res = F.interpolate(res, size = x.shape[2:], mode = 'bilinear', align_corners = False)
        x = x + res
        activation = x

        x = self.ifft_shift(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.output(x)

        return x

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate = learning_rate):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device    
        self.loss_fn = nn.L1Loss()
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr = learning_rate, lr_decay = 1e-6) 
        
        self.log_dir = f"Regression_mod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        try:
            dummy_input = torch.randn(1,1,256,256).to(device)
            self.writer.add_graph(self.model, dummy_input) # use_strict = False
        except Exception as e:
            print(f'TensorBoard graph skipped: {e}')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        max_steps = 300
        total_correct = 0
        total_samples = 0

        progress = tqdm(enumerate(self.train_loader), total = max_steps, desc=f'Epoch {epoch+1}/{epochs}')

        if USE_PROFILER == True:
            with profile(activities=[ProfilerActivity.CPU], schedule = torch.profiler.schedule(
                        wait = 1, warmup = 1, active = 3, repeat = 1),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.log_dir, 'profiler')),
                        record_shapes = True, profile_memory = True, with_stack = True) as prof:

                for batch_idx, (images, labels) in progress:
                    if batch_idx >= max_steps:
                        break

                    self.optimizer.zero_grad()

                    images = images.to(self.device)
                    labels = labels.to(self.device).unsqueeze(1).float()
                    # print(labels)
                    predictions = self.model(images)
                    # print(predictions)
                    loss = self.loss_fn(predictions, labels)

                    loss.backward()
                    self.optimizer.step()

                    total_correct += (torch.abs(predictions - labels) <= 5).sum().item()
                    total_samples += labels.size(0)
                    total_loss += loss.item()

                    prof.step()
        else:
            for batch_idx, (images, labels) in progress:
                if batch_idx >= max_steps:
                    break

                self.optimizer.zero_grad()

                images = images.to(self.device)
                labels = labels.to(self.device).unsqueeze(1).float()
                # print(labels)
                predictions = self.model(images)
                # print(predictions)
                loss = self.loss_fn(predictions, labels)

                loss.backward()
                self.optimizer.step()

                total_correct += (torch.abs(predictions - labels) <= 5).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item()
                            
            return total_loss / max_steps, total_correct / total_samples

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        max_steps = 30
        total_correct = 0
        total_samples = 0

        progress = tqdm(enumerate(self.val_loader), total = max_steps, desc = 'Validation')

        with torch.no_grad():
            for batch_idx, (images, labels) in progress:
                if batch_idx >= max_steps:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device).float()
                labels = labels.unsqueeze(1)
                # print(labels)
                predictions = self.model(images)
                # print(predictions)

                if batch_idx == 0:
                    self.writer.add_histogram('Predictions', predictions.detach().cpu(), epoch)
                    self.writer.add_histogram('Labels', labels.detach().cpu(), epoch)

                loss = self.loss_fn(predictions, labels)
                total_loss += loss.item()

                total_correct += (torch.abs(predictions - labels) <= 5).sum().item()
                total_samples += labels.size(0)

            return total_loss / max_steps, total_correct / total_samples

    def fit(self, epochs, save_path=f'RegressionMod_{mod_name}'):
        best_val_loss = np.inf

        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_epoch(epoch)
            val_loss, val_accuracy = self.validate(epoch)

            print(f'Epoch [{epoch+1}/{epochs}] ', f'Train Accuracy: {train_accuracy}',
                f'Train Loss: {train_loss:.6f}', f'Val Accuracy: {val_accuracy}', f'Val Loss: {val_loss:.6f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            self.writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning Rate', lr, epoch)

        self.writer.flush()
        self.writer.close()

def predict_image(model, image_path, device):
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        prediction = model(image)

    return prediction.item()

##### train and validate model #####
if __name__ == "__main__":
    torch.mps.empty_cache()
    train_loader, val_loader = create_datasets()
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = DefocusRegressionCNN().to(device)

    trainer = Trainer(model=model, train_loader = train_loader, val_loader = val_loader,
        device=device, learning_rate = learning_rate)
    trainer.fit(epochs = epochs, save_path=f'RegressionMod_{mod_name}.pt')

##### test model #####
device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model = DefocusRegressionCNN().to(device)
model.load_state_dict(torch.load('RegressionMod_2.0.pt', map_location = device))
labels = pd.read_csv(csv_file_test)
accuracy_defocus_stig = 0
accuracy_defocus_only = 0

progress = tqdm(enumerate(labels), total = len(labels), desc = 'Test')
for image, label, lx, ly in zip(os.listdir(test_dir), labels['Defocus'], labels['StigX'], labels['StigY']):
    progress.update(1)
    if 'csv' in image:
        continue
    image_path = test_dir + '/' + image
    prediction = predict_image(model, image_path, device)

    if abs(prediction - label) <= 5:
        accuracy_defocus_stig += 1
    
        if lx == 0 and ly == 0:
            accuracy_defocus_only += 1

progress.close()

total_accuracy_defocus_stig = accuracy_defocus_stig/(len(labels))
total_accuracy_defocus = accuracy_defocus_only/(((labels['StigX'] == 0) & (labels['StigY'] == 0)).sum())

print(f'Accuracy for all images: {total_accuracy_defocus_stig}')
print(f'Accuracy for defocused images: {total_accuracy_defocus}')
