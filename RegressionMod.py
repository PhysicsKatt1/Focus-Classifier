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
from torch.utils.data import WeightedRandomSampler
from torch.profiler import profile, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

##### globals #####
path = r'/Users/trentstarkey/Desktop'
image_dir = path + '/RegressionData_30kV_0.09nA'
csv_file = path + '/RegressionData_30kV_0.09nA/labels.csv'

batch = 6
learning_rate = 1e-5
mod_name = '1.0'
epochs = 11

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
    dataset = Data(image_dir = image_dir, csv_file = csv_file, transform = transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(1)
    train_dataset, val_dataset = random_split( dataset, [train_size,val_size], generator = generator)

    train_labels = dataset.data.iloc[train_dataset.indices]['Defocus']
    train_counts = train_labels.value_counts()
    train_weights = train_labels.map(lambda x: 1.0 / np.sqrt(train_counts[x])).values
    sampler_train = WeightedRandomSampler(weights = torch.DoubleTensor(train_weights), num_samples=len(train_dataset), replacement = True)
    
    val_labels = dataset.data.iloc[val_dataset.indices]['Defocus']
    val_counts = val_labels.value_counts()
    val_weights = val_labels.map(lambda x: 1.0 / np.sqrt(val_counts[x])).values
    sampler_val = WeightedRandomSampler(weights = torch.DoubleTensor(val_weights), num_samples=len(val_dataset), replacement = True)

    train_loader = DataLoader(train_dataset, batch_size = batch, sampler = sampler_train)
    val_loader = DataLoader(val_dataset, batch_size = batch, sampler = sampler_val)

    return train_loader, val_loader

class ExpReLU(nn.Module):
       def forward(self, x):
            x = torch.maximum(x * torch.exp(torch.clamp(x, max = 10)), torch.tensor(0., device = x.device))
            return x 

class ResizeResidual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3)

    def forward(self, x, target_size):
        x = self.conv(x)
        x = F.interpolate(x, size = target_size, mode = 'bilinear', align_corners = False)

        return x

class FFTShift(nn.Module):
    def forward(self, x):
        return torch.fft.fftshift(x)

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

        self.pad1 = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(1,128, kernel_size = 3)
        self.bn1 = nn.BatchNorm2d(128)


        self.pad2 = nn.ZeroPad2d(1)
        self.conv2 = nn.Conv2d(128,128, kernel_size = 3)
        self.bn2 = nn.BatchNorm2d(128)


        self.pad3 = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(128,64, kernel_size = 3)
        self.bn3 = nn.BatchNorm2d(64)


        self.pad4 = nn.ZeroPad2d(1)
        self.conv4 = nn.Conv2d(64,8, kernel_size = 3)
        self.bn4 = nn.BatchNorm2d(8)

        self.res1 = nn.Conv2d(128,128, kernel_size = 3)
        self.res2 = nn.Conv2d(128,64, kernel_size = 3)
        self.res3 = nn.Conv2d(64,8, kernel_size = 3)

        self.decoder = nn.ConvTranspose2d(8, 36, kernel_size = 3, padding = 0)
        self.decoder_pool = nn.MaxPool2d(2)
        self.decoder_up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False)
        self.bn5 = nn.BatchNorm2d(36)

        self.res_up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False)
        self.decoder_res = nn.ConvTranspose2d(8, 36, kernel_size = 3, padding = 0)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lin1 = nn.Linear(128, 2048)
        self.lin2 = nn.Linear(2048, 1024)
        self.output = nn.Linear(1024,1)
        self.act = ExpReLU()
        self.patches = Patches(patch_size = 4, resize_x = 200, resize_y = 200)
        self.fft_shift = FFTShift()
        self.ifft_shift = IFFTShift()
        self.patch_conv = nn.Conv2d(in_channels = 576, out_channels = 64, kernel_size = 1)
        self.patch_res = nn.Conv2d(in_channels = 8, out_channels = 64, kernel_size = 1)
        self.fft_conv= nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 1)
        self.fft_res = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 1)
        self.dropout = nn.Dropout(0.05)

    def forward(self,x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.act(x)
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

        x = self.decoder(x)
        x = self.act(x)
        x = self.decoder_pool(x)
        x = self.bn5(x)
        x = self.decoder_up(x)

        res = self.res_up(activation)
        res = self.decoder_res(res)
        res = F.interpolate(res, size = x.shape[2:], mode = 'bilinear', align_corners = False)

        x = x + res

        x = self.patches(x)
        x = self.patch_conv(x)
        x = F.relu(x)

        res = self.patch_res(activation)
        res = F.interpolate(res, size = x.shape[2:], mode = 'bilinear', align_corners = False)

        x = x + res
        activation = x

        x = self.dropout(x)
        x = self.fft_shift(x)
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
    def __init__(self, model, train_loader, val_loader, device, learning_rate = learning_rate):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device    
        self.loss_fn = nn.HuberLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        
        self.log_dir = f"Regression_mod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        try:
            dummy_input = torch.randn(1,1,256,256).to(device)
            self.writer.add_graph(self.model, dummy_input, use_strict = False)
        except Exception as e:
            print(f'TensorBoard graph skipped: {e}')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        max_steps = 60
        total_correct = 0
        total_samples = 0

        progress = tqdm(enumerate(self.train_loader), total = max_steps, desc=f'Epoch {epoch+1}/{epochs}')
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
                            
            return total_loss / max_steps, total_correct / total_samples

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        max_steps = 60
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

if __name__ == "__main__":
    torch.mps.empty_cache()
    train_loader, val_loader = create_datasets()
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = DefocusRegressionCNN().to(device)

    trainer = Trainer(model=model, train_loader = train_loader, val_loader = val_loader,
        device=device, learning_rate = learning_rate)
    trainer.fit(epochs = epochs, save_path=f'RegressionMod_{mod_name}.pt')