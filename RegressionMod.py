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
import torchvision

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
mod_name = '1.0'
epochs = 50
tolerance = 3.0

USE_PROFILER = False

##### define functions #####
def crop_edges(x):
    return x[:, 3:-3, 3:-3]

class PerImageNormalize(object):
    def __call__(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-7)

class MinMaxLabelScaler:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def transform(self, labels):
        return 2.0 * (labels - self.min_val) / (self.max_val - self.min_val + 1e-7) - 1.0

    def inverse_transform(self, scaled_labels):
        return (scaled_labels + 1.0) / 2.0 * (self.max_val - self.min_val) + self.min_val

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
        image_path = os.path.join(self.image_dir, row['Image'] + '.jpeg')
        image = Image.open(image_path).convert('L')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row['Defocus'], dtype=torch.float32)
        return image, label

def get_label_bounds(*csv_files):
    dfs = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(dfs)
    return combined['Defocus'].min(), combined['Defocus'].max()

min_val, max_val = get_label_bounds(csv_file, csv_file_val)

def create_datasets(image_dir, val_dir, csv_file, csv_file_val, batch_size=batch):
    transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Lambda(crop_edges),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            PerImageNormalize()])

    transform_val = transforms.Compose([transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Lambda(crop_edges),
                    PerImageNormalize()])

    generator = torch.Generator().manual_seed(1)

    dataset_train_tf = Data(image_dir=image_dir, csv_file=csv_file, transform=transform_train)
    dataset_val_tf = Data(image_dir=image_dir, csv_file=csv_file, transform=transform_val)

    dataset1_train_tf = Data(image_dir=val_dir, csv_file=csv_file_val, transform=transform_train)
    dataset1_val_tf = Data(image_dir=val_dir, csv_file=csv_file_val, transform=transform_val)

    indices = torch.randperm(len(dataset1_train_tf), generator=generator).tolist()
    train_size = int(0.8 * len(dataset_train_tf))
    val_size = len(dataset_train_tf) - train_size

    split_indices = torch.randperm(len(dataset_train_tf), generator=generator).tolist()
    train_indices0 = split_indices[:train_size]
    val_indices0 = split_indices[train_size:]

    train_dataset0 = Subset(dataset_train_tf, train_indices0)
    val_dataset0 = Subset(dataset_val_tf, val_indices0)

    #secondary train and val split
    train_fraction1 = 0.2
    val_fraction1 = 0.25

    train_size1 = int(train_fraction1 * len(dataset1_train_tf))
    val_size1 = int(val_fraction1 * len(dataset1_train_tf))

    train_indices1 = indices[:train_size1]
    val_indices1 = indices[train_size1:train_size1 + val_size1]

    train_dataset1 = Subset(dataset1_train_tf, train_indices1)
    val_dataset1 = Subset(dataset1_val_tf, val_indices1)

    train_dataset = ConcatDataset([train_dataset0, train_dataset1])
    val_dataset = ConcatDataset([val_dataset0, val_dataset1])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    return train_loader, val_loader

class ExpReLU(nn.Module):
       def forward(self, x):
           x_clamped = torch.clamp(x, max=3.0)

           return F.relu(x_clamped * torch.exp(x_clamped))
            
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
        blur = transforms.GaussianBlur(kernel_size = (3, 3), sigma = (28, 28))
        background = blur(x)
        x = (x - background)
        # max_val = x.amax(dim = (-3, -2, -1), keepdim = True)
        # x = x / max_val    

        x = torch.fft.fft2(x, norm = 'ortho')
        x = torch.fft.fftshift(x)
        x = torch.log1p(torch.abs(x))

        x = x - x.mean(dim=(-2, -1), keepdim=True)
        # fft_transform = transforms.CenterCrop(50)
        # x = fft_transform(x)

        return x

class IFFTShift(nn.Module):
    def forward(self, x):
        return torch.fft.ifftshift(x)

class Patches(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.projection = nn.Conv2d(in_channels = in_channels, out_channels = embed_dim, 
                                    kernel_size = patch_size, stride = patch_size)
        
    def forward(self, x):
        return self.projection(x) 

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, predictions, targets):
        error = torch.abs(targets - predictions).detach() + 1.0
        l1_loss = self.l1(predictions, targets)

        return (l1_loss * error).mean()
    
class DefocusRegressionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = ExpReLU()
        
        self.pad1 = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(1,36, kernel_size = 3)
        self.bn1 = nn.BatchNorm2d(36)

        self.pad_fft = nn.ZeroPad2d(1)
        self.fft = FFT()
        self.fft_conv= nn.Conv2d(in_channels = 36, out_channels = 72, kernel_size = 1)

        self.dropout = nn.Dropout(0.5) 
        
        self.pad2 = nn.ZeroPad2d(1)
        self.conv2 = nn.Conv2d(72,128, kernel_size = 3)
        self.bn2 = nn.BatchNorm2d(128)

        self.ifft_shift = IFFTShift()

        self.fft_res = nn.Conv2d(in_channels=36, out_channels=128, kernel_size=1)

        self.patches = Patches(in_channels = 128, patch_size = 4, embed_dim = 128)
        self.patch_conv = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 1)

        self.patch_res = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)

        self.pad3 = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(64,36, kernel_size = 3)
        self.bn3 = nn.BatchNorm2d(36)

        self.res3 = nn.Conv2d(64, 36, kernel_size = 3)
    
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lin1 = nn.Linear(36, 256)
        self.lin2 = nn.Linear(256, 128)
        self.output = nn.Linear(128,1)
   
    def forward(self,x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.act(x)
        x, indices1 = F.max_pool2d(x, 2, return_indices=True)
        x = self.bn1(x)

        activation = x

        x = self.pad_fft(x)
        x = self.fft(x)
        x = self.fft_conv(x)
        x = self.act(x)

        x = self.dropout(x)

        x = self.pad2(x)
        x = self.conv2(x)
        x = self.act(x)
        x, indices2 = F.max_pool2d(x, 2, return_indices=True)
        x = self.bn2(x)

        x = self.ifft_shift(x)

        res = self.fft_res(activation)
        res = F.interpolate(res, size = x.shape[2:], mode = 'bilinear', align_corners = False)

        x = x + res
        activation = x

        x = self.patches(x)
        x = self.patch_conv(x)
        x = self.act(x)

        res = self.patch_res(activation)
        res = F.interpolate(res, size = x.shape[2:], mode = 'bilinear', align_corners = False)

        x = x + res
        activation = x

        x = self.dropout(x)

        x = self.pad3(x)
        x = self.conv3(x)
        x = self.act(x)
        # x = F.interpolate(x, size=indices1.shape[2:], mode='bilinear', align_corners=False)
        # x = F.max_unpool2d(x, indices=indices1, kernel_size=2)
        x = self.bn3(x)

        res = self.res3(activation)
        res = F.interpolate(res, size = x.shape[2:], mode = 'bilinear', align_corners = False)

        x = x + res
        activation = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = F.relu(self.lin2(x))
        x = self.dropout(x)
        x = self.output(x)

        return x

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, label_scaler, learning_rate = learning_rate):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device    
        self.label_scaler = label_scaler
        self.loss_fn = Loss() 
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr = learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                                    factor=0.5, patience=5, min_lr=1e-6)
        
        self.log_dir = f"Regression_mod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        try:
            dummy_input = torch.randn(1,1,256,256).to(device)
            self.writer.add_graph(self.model, dummy_input) 
        except Exception as e:
            print(f'TensorBoard graph skipped: {e}')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        progress = tqdm(enumerate(self.train_loader), total = len(self.train_loader), desc=f'Training {epoch+1}/{epochs}')

        if USE_PROFILER == True:
            with profile(activities=[ProfilerActivity.CPU], schedule = torch.profiler.schedule(
                        wait = 1, warmup = 1, active = 3, repeat = 1),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.log_dir, 'profiler')),
                        record_shapes = True, profile_memory = True, with_stack = True) as prof:


                for i, (images, labels) in progress:
                    images = images.to(self.device)
                    raw_labels = labels.to(self.device).unsqueeze(1)

                    scaled_labels = self.label_scaler.transform(raw_labels)

                    self.optimizer.zero_grad()
                    predictions = self.model(images)
                    loss = self.loss_fn(predictions, scaled_labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    predictions_raw = self.label_scaler.inverse_transform(predictions)
                    total_correct += (torch.abs(predictions_raw - raw_labels) <= tolerance).sum().item()
                    total_samples += raw_labels.size(0)
                    total_loss += loss.item()
                    prof.step()

                return total_loss / len(self.train_loader), total_correct / total_samples

        else:
            for i, (images, labels) in progress:
                images = images.to(self.device)
                raw_labels = labels.to(self.device).unsqueeze(1)

                scaled_labels = self.label_scaler.transform(raw_labels)

                self.optimizer.zero_grad()
                predictions = self.model(images)
                loss = self.loss_fn(predictions, scaled_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                predictions_raw = self.label_scaler.inverse_transform(predictions)
                total_correct += (torch.abs(predictions_raw - raw_labels) <= tolerance).sum().item()
                total_samples += raw_labels.size(0)
                total_loss += loss.item()

            return total_loss / len(self.train_loader), total_correct / total_samples

    def validate(self, epoch):
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        progress = tqdm(enumerate(self.val_loader), total = len(self.val_loader), desc=f'Validation {epoch+1}/{epochs}')

        if USE_PROFILER == True:
            with torch.no_grad():
                for i, (images, labels) in progress:
                    images = images.to(self.device)
                    raw_labels = labels.to(self.device).unsqueeze(1)

                    scaled_labels = self.label_scaler.transform(raw_labels)
                    predictions = self.model(images)
                    loss = self.loss_fn(predictions, scaled_labels)

                    predictions_raw = self.label_scaler.inverse_transform(predictions)
                    total_correct += (torch.abs(predictions_raw - raw_labels) <= tolerance).sum().item()
                    total_samples += raw_labels.size(0)
                    total_loss += loss.item()

                return total_loss / len(self.val_loader), total_correct / total_samples                 

        else:
            with torch.no_grad():
                for i, (images, labels) in progress:
                    images = images.to(self.device)
                    raw_labels = labels.to(self.device).unsqueeze(1)

                    scaled_labels = self.label_scaler.transform(raw_labels)
                    predictions = self.model(images)
                    loss = self.loss_fn(predictions, scaled_labels)

                    predictions_raw = self.label_scaler.inverse_transform(predictions)
                    total_correct += (torch.abs(predictions_raw - raw_labels) <= tolerance).sum().item()
                    total_samples += raw_labels.size(0)
                    total_loss += loss.item()

            return total_loss / len(self.val_loader), total_correct / total_samples

    def fit(self, epochs, save_path=f'RegressionMod_{mod_name}'):
        best_val_loss = np.inf

        if USE_PROFILER == True:
            for epoch in range(epochs):
                train_loss, train_accuracy = self.train_epoch(epoch)
                val_loss, val_accuracy = self.validate(epoch)
                self.scheduler.step(val_loss)

                print(f'Epoch [{epoch+1}/{epochs}] ', f'Train Accuracy: {train_accuracy}',
                    f'Train Loss: {train_loss:.6f}', f'Val Accuracy: {val_accuracy}', f'Val Loss: {val_loss:.6f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({'model_state_dict': self.model.state_dict(), 
                                'label_min': self.label_scaler.min_val,
                                'label_max': self.label_scaler.max_val}, save_path)

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
                # self.scheduler.step(val_loss)

                print(f'Epoch [{epoch+1}/{epochs}] ', f'Train Accuracy: {train_accuracy}',
                    f'Train Loss: {train_loss:.6f}', f'Val Accuracy: {val_accuracy}', f'Val Loss: {val_loss:.6f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({'model_state_dict': self.model.state_dict(), 
                                'label_min': self.label_scaler.min_val,
                                'label_max': self.label_scaler.max_val}, save_path)

        return 

class TestData(Dataset):
    def __init__(self, test_dir, csv_file, transform=None):
        self.test_dir = test_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = str(int(row['Image']))
        image_path = os.path.join(self.test_dir, image_name + '.jpeg')
        image = Image.open(image_path).convert('L')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row['Defocus'],dtype=torch.float32)

        return image, label

def predict_image(model, test_loader, device, label_scaler, tolerance):
    model.eval()
    correct = 0
    samples = 0

    progress = tqdm(test_loader, desc='Test')

    with torch.no_grad():
        for images, labels in progress:
            images = images.to(device)
            raw_labels = labels.to(device).unsqueeze(1).float()

            predictions = model(images)

            predictions_raw = label_scaler.inverse_transform(predictions)

            hit = torch.abs(predictions_raw - raw_labels) <= tolerance
            correct += hit.sum().item()
            samples += raw_labels.size(0)

    total_accuracy = correct / samples

    return total_accuracy

##### train and validate model #####
if __name__ == "__main__":

    torch.mps.empty_cache()
    min_val, max_val = get_label_bounds(csv_file)
    label_scaler = MinMaxLabelScaler(min_val=min_val, max_val=max_val)

    train_loader, val_loader = create_datasets(image_dir, val_dir, csv_file, csv_file_val)
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = DefocusRegressionCNN().to(device)

    trainer = Trainer(model=model, train_loader = train_loader, val_loader = val_loader,
        device=device,  label_scaler =  label_scaler, learning_rate = learning_rate)
    trainer.fit(epochs = epochs, save_path=f'RegressionMod_{mod_name}.pt')

    ##### test model #####
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = DefocusRegressionCNN().to(device)
    checkpoint = torch.load('RegressionMod_1.0.pt', map_location = device, weights_only = False)
    model.load_state_dict(checkpoint['model_state_dict'])

    label_scaler = MinMaxLabelScaler(min_val=checkpoint['label_min'], max_val=checkpoint['label_max'])

    transform_test = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                                         transforms.Lambda(crop_edges), PerImageNormalize()])
    test_dataset = TestData(test_dir = test_dir, csv_file = csv_file_test, transform = transform_test)
    test_loader = DataLoader(test_dataset, batch_size = batch, shuffle = False)

    total_accuracy = predict_image(model, test_loader, device, label_scaler, tolerance)
    print(f'Total accuracy: {total_accuracy}')

    
