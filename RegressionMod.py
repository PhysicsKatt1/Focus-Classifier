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
from torch.utils.data import ConcatDataset, Subset
from torch.profiler import profile, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

##### globals #####
path = r'/Users/trentstarkey/Desktop'
image_dir = path + '/RegressionData_30kV_0.09nA'
val_dir = path + '/RegressionData_30kV_0.09nA_val'
csv_file = path + '/RegressionData_30kV_0.09nA/labels.csv'
csv_file_val = path + '/RegressionData_30kV_0.09nA_val/labels.csv'
test_dir = path + '/RegressionData_30kV_0.09nA_test'
csv_file_test = path + '/RegressionData_30kV_0.09nA_test/labels.csv'

batch = 32
learning_rate = 1e-3
mod_name = '2.2'
epochs = 12
tolerance = 3.0

USE_PROFILER = False

##### define functions #####
class Data(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
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
    transform_train = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    transform_val = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    generator = torch.Generator().manual_seed(1)
    
    #----- create datasets -----#
    dataset = Data(image_dir=image_dir, csv_file=csv_file, transform=transform_train)
    dataset1 = Data(image_dir=val_dir, csv_file=csv_file_val, transform=transform_val)
    indices = torch.randperm(len(dataset1), generator = generator).tolist()
    
    #----- create training and val split -----#
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset0, val_dataset0 = random_split(dataset, [train_size, val_size], generator = generator)
    
    #----- create secondary training and val split -----#
    train_fraction1 = 0.2
    val_fraction1 = 0.25
    
    train_size1 = int(train_fraction1 * len(dataset1))
    val_size1 = int(val_fraction1 * len(dataset1))

    train_indices1 = indices[:train_size1]
    val_indices1 = indices[train_size1:train_size1 + val_size1]
    train_dataset1 = Subset(dataset1, train_indices1)
    val_dataset1 = Subset(dataset1, val_indices1)
    
    val_dataset = ConcatDataset([val_dataset0, val_dataset1])
    train_dataset = ConcatDataset([train_dataset0, train_dataset1])

    #----- create final dataloaders -----#
    train_loader = DataLoader(train_dataset, batch_size = batch)
    val_loader = DataLoader(val_dataset, batch_size = batch, shuffle = False)

    return train_loader, val_loader

def get_normalization_stats(train_loader):
    image_sum = 0.0
    image_squared_sum = 0.0
    image_count = 0

    label_sum = 0.0
    label_squared_sum = 0.0
    label_count = 0

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.float()
            labels = labels.float()

            image_sum += images.sum().item()
            image_squared_sum += (images ** 2).sum().item()
            image_count += images.numel()

            label_sum += labels.sum().item()
            label_squared_sum += (labels ** 2).sum().item()
            label_count += labels.numel()

    image_mean = image_sum / image_count
    image_std = np.sqrt(image_squared_sum / image_count - image_mean ** 2)
    label_mean = label_sum / label_count
    label_std = np.sqrt(label_squared_sum / label_count - label_mean ** 2)

    return image_mean, image_std, label_mean, label_std

class ExpReLU(nn.Module):
       def forward(self, x):
            x = torch.clamp(x, max = 5) 
            x = torch.clamp_min(x * torch.exp(x), 0)
            return x 

class ResizeResidual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3)

    def forward(self, x, target_size):
        x = self.conv(x)
        x = F.interpolate(x, size = target_size, mode = 'bilinear', align_corners = False)

        return x

class FFT(nn.Module):
    def forward(self, x):
        # x  = x[:, :, 3:-3, 3:-3]
        # blur = torchvision.transforms.GaussianBlur(kernel_size = (1, 1), sigma = (28, 28))
        # background = blur(x)
        # x = (x - background)
        # x = x / torch.max(x)

        # x = torch.fft.fft2(x, norm = 'ortho')
        # x = torch.fft.fftshift(x)
        # x = torch.log1p(torch.abs(x))
        # x = x - (torch.mean(x, dim=(-2, -1), keepdim=True) - 1e-05)

        # h, w = x.shape[-2:]
        # hc, wc = h//2, w//2 
        # x = x[:, :, hc - 128: hc + 128, wc - 128: wc + 128] 
        return torch.fft.fftshift(x)
    
class IFFTShift(nn.Module):
    def forward(self, x):
        return torch.fft.ifftshift(x)

class Patches(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim ):
        super().__init__()
        self.projection = nn.Conv2d(in_channels = in_channels, out_channels = embed_dim, 
                                    kernel_size=patch_size, stride = patch_size)
        
    def forward(self, x):
        return self.projection(x) 

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, predictions, targets):
        error = torch.abs(targets - predictions)
        var = torch.var(predictions)
        l1_loss = self.l1(predictions * torch.exp(error * var), targets)

        return l1_loss 
    
class DefocusRegressionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = ExpReLU()
        
        self.pad1 = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(1,128, kernel_size = 3)
        self.bn1 = nn.BatchNorm2d(128)

        self.pad2 = nn.ZeroPad2d(1)
        self.conv2 = nn.Conv2d(128,128, kernel_size = 3)
        self.bn2 = nn.BatchNorm2d(128)

        self.res1 = nn.Conv2d(128,128, kernel_size = 3)

        self.pad3 = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(128,64, kernel_size = 3)
        self.bn3 = nn.BatchNorm2d(64)

        self.res2 = nn.Conv2d(128,64, kernel_size = 3)
        
        self.pad4 = nn.ZeroPad2d(1)
        self.conv4 = nn.Conv2d(64,8, kernel_size = 3)
        self.bn4 = nn.BatchNorm2d(8)

        self.res3 = nn.Conv2d(64,8, kernel_size = 3)

        self.patches = Patches(in_channels = 8, patch_size = 4, embed_dim = 128)
        self.patch_conv = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 1)

        self.patch_res = nn.Conv2d(in_channels = 8, out_channels = 64, kernel_size = 1)
       
        self.dropout = nn.Dropout(0.05)
        self.fft = FFT()
        self.fft_conv= nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 1)

        self.fft_res = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 1)

        self.ifft_shift = IFFTShift()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lin1 = nn.Linear(128, 2048)
        self.lin2 = nn.Linear(2048, 1024)
        self.output = nn.Linear(1024,1)
   
    def forward(self,x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.act(x)
        # print((x == 0).float().mean().item())
        x = F.max_pool2d(x,2)
        x = self.bn1(x)

        activation = x

        x = self.pad2(x)
        x = self.conv2(x)
        x = self.act(x)
        x = F.max_pool2d(x,2)
        x = self.bn2(x)

        res = self.res1(activation)
        res = F.interpolate(res, size = x.shape[2:], mode = 'bilinear', align_corners = False)

        x = x + res
        activation = x

        x = self.pad3(x)
        x = self.conv3(x)
        x = self.act(x)
        x = F.max_pool2d(x,2)
        x = self.bn3(x)

        res = self.res2(activation)
        res = F.interpolate(res, size = x.shape[2:], mode = 'bilinear', align_corners=False)

        x = x + res
        activation = x

        x = self.pad4(x)
        x = self.conv4(x)
        x = self.act(x)
        x = F.max_pool2d(x,2)
        x = self.bn4(x)

        res = self.res3(activation)
        res = F.interpolate(res, size = x.shape[2:], mode = 'bilinear', align_corners = False)

        x = x + res
        activation = x

        x = self.patches(x)
        x = self.patch_conv(x)
        x = F.relu(x)

        res = self.patch_res(activation)
        res = F.interpolate(res, size = x.shape[2:], mode = 'bilinear', align_corners = False)

        x = x + res
        activation = x

        x = self.dropout(x)
        x = self.fft(x)
        x = self.fft_conv(x)
        x = F.relu(x)

        res = self.fft_res(activation)
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
    def __init__(self, model, train_loader, val_loader, device, 
                 image_std, image_mean, label_std, label_mean, learning_rate = learning_rate):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device    
        self.image_std = image_std
        self.image_mean = image_mean
        self.label_std = label_std
        self.label_mean = label_mean
        self.loss_fn = Loss()
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=learning_rate) 
        
        self.log_dir = f"Regression_mod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        try:
            dummy_input = torch.randn(1,1,256,256).to(device)
            self.writer.add_graph(self.model, dummy_input) 
        except Exception as e:
            print(f'TensorBoard graph skipped: {e}')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_batches = 0

        progress = tqdm(enumerate(self.train_loader), total = len(self.train_loader), desc=f'Training {epoch+1}/{epochs}')

        if USE_PROFILER == True:
            with profile(activities=[ProfilerActivity.CPU], schedule = torch.profiler.schedule(
                        wait = 1, warmup = 1, active = 3, repeat = 1),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.log_dir, 'profiler')),
                        record_shapes = True, profile_memory = True, with_stack = True) as prof:

                for batch_idx, (images, labels) in progress:
                    self.optimizer.zero_grad()

                    images = images.to(self.device)
                    labels = labels.to(self.device).unsqueeze(1).float()
                    images = (images - self.image_mean) / self.image_std
                    labels = (labels - self.label_mean) / self.label_std
                    # print(labels)
                    predictions = self.model(images)
                    # print(predictions)
                    loss = self.loss_fn(predictions, labels)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    predictions_raw = predictions * self.label_std + self.label_mean
                    labels_raw = labels * self.label_std + self.label_mean

                    total_correct += (torch.abs(predictions_raw - labels_raw) <= tolerance).sum().item()
                    total_samples += labels.size(0)
                    total_loss += loss.item()
                    total_batches += 1

                    prof.step()
            return total_loss / total_batches, total_correct / total_samples

        else:
            for batch_idx, (images, labels) in progress:
                self.optimizer.zero_grad()

                images = images.to(self.device)
                labels = labels.to(self.device).unsqueeze(1).float()
                images = (images - self.image_mean) / self.image_std
                labels = (labels - self.label_mean) / self.label_std
                # print(labels)
                predictions = self.model(images)
                # print(predictions)
                loss = self.loss_fn(predictions, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                predictions_raw = predictions * self.label_std + self.label_mean
                labels_raw = labels * self.label_std + self.label_mean

                total_correct += (torch.abs(predictions_raw - labels_raw) <= tolerance).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item()
                total_batches += 1
                            
            return total_loss / total_batches, total_correct / total_samples

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_batches = 0

        progress = tqdm(enumerate(self.val_loader), total = len(self.val_loader), desc = 'Validation') 

        if USE_PROFILER == True:
            with torch.no_grad():
                for batch_idx, (images, labels) in progress:
                    images = images.to(self.device)
                    labels = labels.to(self.device).float()
                    labels = labels.unsqueeze(1)
                    images = (images - self.image_mean) / self.image_std
                    labels = (labels - self.label_mean) / self.label_std

                    # print(labels)
                    predictions = self.model(images)
                    # print(predictions)

                    if batch_idx == 0:
                        self.writer.add_histogram('Predictions', predictions.detach().cpu(), epoch)
                        self.writer.add_histogram('Labels', labels.detach().cpu(), epoch)

                    loss = self.loss_fn(predictions, labels)
                    total_loss += loss.item()

                    predictions_raw = predictions * self.label_std + self.label_mean
                    labels_raw = labels * self.label_std + self.label_mean

                    total_correct += (torch.abs(predictions_raw - labels_raw) <= tolerance).sum().item()
                    total_samples += labels.size(0)
                    total_batches += 1

            return total_loss / total_batches, total_correct / total_samples

        else:
            with torch.no_grad():
                for batch_idx, (images, labels) in progress:
                    images = images.to(self.device)
                    labels = labels.to(self.device).float()
                    labels = labels.unsqueeze(1)
                    images = (images - self.image_mean) / self.image_std
                    labels = (labels - self.label_mean) / self.label_std
                    # print(labels)
                    predictions = self.model(images)
                    # print(predictions)

                    loss = self.loss_fn(predictions, labels)
                    total_loss += loss.item()

                    predictions_raw = predictions * self.label_std + self.label_mean
                    labels_raw = labels * self.label_std + self.label_mean

                    # print(predictions_raw,labels_raw)

                    total_correct += (torch.abs(predictions_raw - labels_raw) <= tolerance).sum().item()
                    total_samples += labels.size(0)
                    total_batches += 1

            return total_loss / total_batches, total_correct / total_samples

    def fit(self, epochs, save_path=f'RegressionMod_{mod_name}'):
        best_val_loss = np.inf

        if USE_PROFILER == True:
            for epoch in range(epochs):
                train_loss, train_accuracy = self.train_epoch(epoch)
                val_loss, val_accuracy = self.validate(epoch)

                print(f'Epoch [{epoch+1}/{epochs}] ', f'Train Accuracy: {train_accuracy}',
                    f'Train Loss: {train_loss:.6f}', f'Val Accuracy: {val_accuracy}', f'Val Loss: {val_loss:.6f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({'model_state_dict': self.model.state_dict(),
                                    'image_mean': self.image_mean, 'image_std': self.image_std,
                                    'label_mean': self.label_mean, 'label_std': self.label_std}, save_path)

                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/validation', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_accuracy, epoch)
                self.writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning Rate', lr, epoch)

            self.writer.flush()
            self.writer.close()

            return 

        else:
             for epoch in range(epochs):
                train_loss, train_accuracy = self.train_epoch(epoch)
                val_loss, val_accuracy = self.validate(epoch)

                print(f'Epoch [{epoch+1}/{epochs}] ', f'Train Accuracy: {train_accuracy}',
                    f'Train Loss: {train_loss:.6f}', f'Val Accuracy: {val_accuracy}', f'Val Loss: {val_loss:.6f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({'model_state_dict': self.model.state_dict(),
                                    'image_mean': self.image_mean, 'image_std': self.image_std,
                                    'label_mean': self.label_mean, 'label_std': self.label_std}, save_path)

        return 

def predict_image(model, image_path, device, image_mean, image_std, label_mean,label_std):
    transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)
    image = (image - image_mean) / image_std

    model.eval()

    with torch.no_grad():
        prediction = model(image)

    prediction = prediction * label_std + label_mean

    return prediction.item()

##### train and validate model #####
if __name__ == "__main__":

    torch.mps.empty_cache()
    train_loader, val_loader = create_datasets()
    image_mean, image_std, label_mean, label_std = get_normalization_stats(train_loader)
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = DefocusRegressionCNN().to(device)

    trainer = Trainer(model=model, train_loader = train_loader, val_loader = val_loader,
        device=device, image_mean = image_mean, image_std = image_std, 
        label_mean = label_mean, label_std = label_std,  learning_rate = learning_rate)
    trainer.fit(epochs = epochs, save_path=f'RegressionMod_{mod_name}.pt')

    ##### test model #####
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = DefocusRegressionCNN().to(device)
    checkpoint = torch.load('RegressionMod_2.2.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    image_mean = checkpoint['image_mean']
    image_std = checkpoint['image_std']
    label_mean = checkpoint['label_mean']
    label_std = checkpoint['label_std']

    labels = pd.read_csv(csv_file_test)
    accuracy_defocus_stig = 0
    accuracy_defocus_only = 0

    progress = tqdm(labels.iterrows(), total=len(labels), desc='Test')

    for _, row in progress:
        image = str(int(row['Image']))
        image_path = test_dir + '/' + image + '.jpeg'
        label = row['Defocus']
        lx = row['StigX']
        ly = row['StigY']

        prediction = predict_image(model, image_path, device, image_mean, image_std, label_mean, label_std)

        if abs(prediction - label) <= tolerance:
            accuracy_defocus_stig += 1

            if lx == 0 and ly == 0:
                accuracy_defocus_only += 1

    progress.close()

    total_accuracy_defocus_stig = accuracy_defocus_stig / len(labels)
    total_accuracy_defocus = accuracy_defocus_only / (((labels['StigX'] == 0) & (labels['StigY'] == 0)).sum())

    print(f'Accuracy for all images: {total_accuracy_defocus_stig}')
    print(f'Accuracy for defocused images: {total_accuracy_defocus}')
