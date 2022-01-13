import pandas as pd
from PIL import Image
from ppc import get_train_val_test_dfs, red_count_preprocess
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, utils
import pytorch_lightning as pl
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, transform
import math
from pytorch_lightning.loggers import MLFlowLogger
seed_everything(13)

fpath2img = {}

class KORLData(Dataset):
    def __init__(self, X, y, markers, transform=None):
        self.X = X
        self.y = y
        self.markers = list(markers)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imgs = []
        for i in self.markers:
            fpath = self.X.iloc[idx][f"marker{i}"]
            if fpath not in fpath2img:
                fpath2img[fpath] = self.transform(Image.open(fpath))
            imgs.append(fpath2img[fpath])
        if len(imgs) == 1:
            sample = np.concatenate([np.expand_dims(np.array(imgs[0])[:,:,j], axis=0) for j in range(3)], axis=0) / 255.
        else:
            sample = np.concatenate([np.expand_dims(np.array(img)[:,:,0], axis=0) for img in imgs], axis=0) / 255.
        return torch.FloatTensor(sample), self.y[idx]

class KORLDataModule(pl.LightningDataModule):
    def __init__(self, markers=[1, 2, 3, 4, 5, 6], batch_size=16, img_resize=256, balance_dataset=False):
        super().__init__()
        self.batch_size = batch_size
        self.img_resize = img_resize
        self.markers = list(markers)
        self.balance_dataset = balance_dataset

    def load_data(self):
        return get_train_val_test_dfs(val_size=.3)

    def prepare_data(self):
        df_train, df_val, _ = self.load_data()
        #
        to_keep = ['patient'] + [f"marker{m}" for m in self.markers] + ['OS', 'target']
        df_train = df_train[to_keep].drop_duplicates().reset_index(drop=True)
        df_val = df_val[to_keep].drop_duplicates().reset_index(drop=True)
        #
        if self.balance_dataset:
            min_class_card = df_train['target'].value_counts().min()
            df_train = df_train.groupby('target').apply(lambda x: x.sample(min_class_card))
            df_train.index = df_train.index.droplevel(0)
        df_train = df_train.sample(frac=1.).reset_index(drop=True)
        self.X_train = df_train
        self.y_train = df_train['target'].apply(int).values
        self.X_val = df_val
        self.y_val = df_val['target'].apply(int).values

    def setup(self, stage=None):
        self.ds_train = KORLData(self.X_train, self.y_train, self.markers,
                                 transform=transforms.Resize(self.img_resize))
        self.ds_val = KORLData(self.X_val, self.y_val, self.markers,
                               transform=transforms.Resize(self.img_resize))

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True,
                         num_workers=min(mp.cpu_count(), 2*torch.cuda.device_count()))
    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=True,
                         num_workers=min(mp.cpu_count(), 2*torch.cuda.device_count()))

    def test_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=True,
                         num_workers=min(mp.cpu_count(), 2*torch.cuda.device_count()))


def compute_accuracy(out, y):
    if y.size(0) == 0:
        return 0.
    pred = F.softmax(out, 1)
    unk_idx = y.max().item() + 1
    _max = torch.max(pred, 1).values
    _pred = torch.argmax(pred, 1)
    acc = (_pred.to('cpu') == y.to('cpu')).sum().item() / y.size(0)
    return acc


class BaseModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, l1=0., l2=0., weight=None):
        super().__init__()
        self.lr = learning_rate
        self.l1 = l1
        self.l2 = l2
        self.weight = weight if weight is None else torch.tensor(weight, dtype=torch.float, device=self.device)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        self.weight = self.weight if self.weight is None else self.weight.to(self.device)
        loss = F.cross_entropy(out, y.view(-1), weight=self.weight)
        # L1
        if self.l1 > 0:
            l1_reg = 0.
            for param in self.parameters():
                l1_reg += (param.abs() * self.l1).sum()
            loss += l1_reg
        # L2
        if self.l2 > 0.:
            l2_reg = 0.
            for param in self.parameters():
                l2_reg += (param.pow(2) * self.l2).sum()
            loss += l2_reg
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        # Accuracy
        accuracy = compute_accuracy(out, y.view(-1))
        self.log("train_acc", accuracy, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        self.weight = self.weight if self.weight is None else self.weight.to(self.device)
        loss = F.cross_entropy(out, y.view(-1), weight=self.weight)
        self.log("val_loss", loss, prog_bar=True)
        accuracy = compute_accuracy(out, y.view(-1))
        self.log("val_acc", accuracy, prog_bar=True)
        return {"loss": loss, "accuracy": accuracy, "n_samples": x.size(0)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        self.weight = self.weight if self.weight is None else self.weight.to(self.device)
        loss = F.cross_entropy(out, y.view(-1), weight=self.weight)
        self.log("test_loss", loss, prog_bar=True)
        accuracy = compute_accuracy(out, y.view(-1))
        self.log("test_acc", accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer



class KORLModel(BaseModel):
    def __init__(self, learning_rate=1e-3, l1=0., l2=0., weight=None):
        super().__init__(learning_rate, l1, l2, weight)
        # Different image sizes
        # 128, 170 -> 18,29
        # 256, 341 -> 50,71
        # 300, 400
        # 512, 682
        self.conv1 = nn.Conv2d(6, 3, kernel_size=7) # 128x170 -> 122x164
        self.pool1 = nn.MaxPool2d(2) # 122x164 -> 61x82
        self.conv2 = nn.Conv2d(3, 1, kernel_size=5) # 61x82 -> 57x78
        self.conv3 = nn.Conv2d(1, 1, kernel_size=5) # 57x78 -> 53x74
        self.pool2 = nn.MaxPool2d(2) # 53x74 -> 26x37
        self.conv4 = nn.Conv2d(1, 1, kernel_size=5) # 26x37 -> 22x33
        self.conv5 = nn.Conv2d(1, 1, kernel_size=5) # 22x33 -> 18x29
        self.lin_out = nn.Linear(50*71, 3)

    def forward(self, x):
        # Shape (N, 6, 256, 341)
        out = F.relu(self.conv1(x))
        # Shape (N, 3, 250, 335)
        out = self.pool1(out)
        # Shape (N, 3, 125, 167)
        out = F.relu(self.conv2(out))
        # Shape (N, 1, 121, 163)
        out = F.relu(self.conv3(out))
        # Shape (N, 1, 117, 159)
        out = self.pool2(out)
        # Shape (N, 1, 58, 79)
        out = F.relu(self.conv4(out))
        # Shape (N, 1, 54, 75)
        out = F.relu(self.conv5(out))
        # Shape (N, 1, 50, 71)
        out = self.lin_out(out.view(out.size(0), -1))
        # Shape (N, 3)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super(Bottleneck, self).__init__()
        # print('Creating bottleneck with kernel size {} and stride {} with padding {}'.format(kernel_size, stride, (kernel_size - 1) // 2))
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=0, bias=False) # changed padding from (kernel_size - 1) // 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:,:,:-diff,:-diff]
        out += residual
        out = self.relu(out)
        return out

class KORLBagNet(BaseModel):
    def __init__(self,
                 block, layers, strides=[1, 2, 2, 2], kernel3=[0, 0, 0, 0],
                 num_classes=1000, avg_pool=True,
                 learning_rate=1e-3, l1=0., l2=0., weight=None):
        super().__init__(learning_rate, l1, l2, weight)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], kernel3=kernel3[0], prefix='layer1')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], kernel3=kernel3[1], prefix='layer2')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], kernel3=kernel3[2], prefix='layer3')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], kernel3=kernel3[3], prefix='layer4')
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avg_pool = avg_pool
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0, prefix=''):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        # (N, 3, 256, 256)
        x = self.conv1(x)
        # (N, 64, 256, 256)
        x = self.conv2(x)
        # (N, 64, 254, 254)
        x = self.bn1(x)
        # (N, 64, 254, 254)
        x = self.relu(x)
        # (N, 64, 254, 254)

        x = self.layer1(x)
        # (N, 256, 126, 126) # pool and increase channels
        x = self.layer2(x)
        # (N, 512, 62, 62)
        x = self.layer3(x)
        # (N, 1024, 30, 30)
        x = self.layer4(x)
        # (N, 2048, 30, 30)

        if self.avg_pool:
            x = nn.AvgPool2d(x.size()[2], stride=1)(x)
            # (N, 2048, 1, 1) # should be 1 at the end, square
            x = x.view(x.size(0), -1)
            # (N, 2048)
            x = self.fc(x) #
        else:
            x = x.permute(0,2,3,1)
            x = self.fc(x)

        return x


def train_basic_model(batch_size=512, learning_rate=2e-5, img_size=256,
                      exp_name='Basic', tracking_uri='./mlruns',
                      max_epochs=2, markers=[1, 2, 3, 4, 5, 6],
                      balance_dataset=True):
    data_module = KORLDataModule(batch_size=batch_size,
                                 img_resize=img_size,
                                 markers=markers,
                                 balance_dataset=balance_dataset)
    data_module.prepare_data()
    data_module.setup()

    model = KORLModel(learning_rate=learning_rate)
    mlf_logger = MLFlowLogger(experiment_name=exp_name,
                              tracking_uri=tracking_uri)
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=max_epochs,
                         val_check_interval=0.2,
                         logger=mlf_logger)
    trainer.fit(model, data_module)
    return model.to('cpu')


def train_bagnet_model(batch_size=128, learning_rate=2e-4, img_size=(64, 64),
                      exp_name='BG', tracking_uri='./mlruns',
                      max_epochs=5, markers=[1]):
    data_module = KORLDataModule(batch_size=batch_size, img_resize=img_size,
                                 balance_dataset=False,
                                 markers=markers)
    data_module.prepare_data()
    data_module.setup()

    model = KORLBagNet(Bottleneck, [3, 4, 6, 3], strides=[2, 2, 2, 1],
                       kernel3=[1, 1, 1, 0], learning_rate=learning_rate,
                       num_classes=3)
    mlf_logger = MLFlowLogger(experiment_name=exp_name,
                              tracking_uri=tracking_uri)
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=max_epochs,
                         val_check_interval=0.2,
                         logger=mlf_logger)
    trainer.fit(model, data_module)
    return model.to('cpu')


def predict(model, df, img_size=(64, 64), markers=[1, 2, 3, 4, 5, 6], batch_size=16):
    # Create dataloader
    dl = DataLoader(KORLData(df,
                             df['target'].apply(int).values if 'target' in df.columns else np.zeros(len(df)),
                             markers,
                             transform=transforms.Resize(img_size)),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=min(mp.cpu_count(), 2*torch.cuda.device_count()))
    # Predict
    if torch.cuda.is_available():
        model = model.to('cuda')
    model.eval()
    predictions = np.array([])
    for x, y in dl:
        with torch.no_grad():
            if torch.cuda.is_available():
                x = x.to('cuda')
            out = model(x)
            pred = F.softmax(out.cpu(), 1)
            _max = torch.max(pred, 1).values
            _pred = torch.argmax(pred, 1)
            predictions = np.concatenate([predictions, _pred.numpy()], axis=0)
    return predictions


def compute_aggregated_accuracy(df):
    assert 'target' in df.columns and 'pred' in df.columns and 'patient' in df.columns
    total, correct = 0, 0
    for idx, gb_df in df.groupby('patient'):
        patient_pred = round(gb_df['pred'].mean())
        patient_pred = gb_df['pred'].value_counts().index.tolist()[0]
        correct += (patient_pred == gb_df.iloc[0]['target']) * 1
        total += 1
    return correct / total if total != 0 else 0.


def generate_heatmap_pytorch(model, image, target, patchsize, image_size):
    """
    Generates high-resolution heatmap for a BagNet by decomposing the
    image into all possible patches and by computing the logits for
    each patch.

    Parameters
    ----------
    model : Pytorch Model
        This should be one of the BagNets.
    image : Numpy array of shape [1, 3, X, X]
        The image for which we want to compute the heatmap.
    target : int
        Class for which the heatmap is computed.
    patchsize : int
        The size of the receptive field of the given BagNet.

    """
    with torch.no_grad():
        model.eval()
        # pad with zeros
        _, c, x, y = image.shape
        padded_image = np.zeros((c, x + patchsize - 1, y + patchsize - 1))
        padded_image[:, (patchsize-1)//2:(patchsize-1)//2 + x, (patchsize-1)//2:(patchsize-1)//2 + y] = image[0]
        image = padded_image[None].astype(np.float32)

        # turn to torch tensor
        input = torch.from_numpy(image).cuda()

        # extract patches
        patches = input.permute(0, 2, 3, 1)
        patches = patches.unfold(1, patchsize, 1).unfold(2, patchsize, 1)
        num_rows = patches.shape[1]
        num_cols = patches.shape[2]
        patches = patches.contiguous().view((-1, 3, patchsize, patchsize))

        # compute logits for each patch
        logits_list = []

        for batch_patches in torch.split(patches, 1000):
            #print(batch_patches.size())
            logits = model(batch_patches)
            logits = logits[:, target]#[:, 0]
            logits_list.append(logits.data.cpu().numpy().copy())

        logits = np.hstack(logits_list)
        return logits.reshape(image_size)


def plot_heatmap(heatmap, original, ax, cmap='RdBu_r',
                 percentile=99, dilation=0.5, alpha=0.25):
    """
    Plots the heatmap on top of the original image
    (which is shown by most important edges).

    Parameters
    ----------
    heatmap : Numpy Array of shape [X, X]
        Heatmap to visualise.
    original : Numpy array of shape [X, X, 3]
        Original image for which the heatmap was computed.
    ax : Matplotlib axis
        Axis onto which the heatmap should be plotted.
    cmap : Matplotlib color map
        Color map for the visualisation of the heatmaps (default: RdBu_r)
    percentile : float between 0 and 100 (default: 99)
        Extreme values outside of the percentile range are clipped.
        This avoids that a single outlier dominates the whole heatmap.
    dilation : float
        Resizing of the original image. Influences the edge detector and
        thus the image overlay.
    alpha : float in [0, 1]
        Opacity of the overlay image.

    """
    if len(heatmap.shape) == 3:
        heatmap = np.mean(heatmap, 0)

    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, heatmap.shape[1], dx)
    yy = np.arange(0.0, heatmap.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_original = plt.get_cmap('Greys_r')
    cmap_original.set_bad(alpha=0)
    overlay = None
    if original is not None:
        # Compute edges (to overlay to heatmaps later)
        original_greyscale = original if len(original.shape) == 2 else np.mean(original, axis=-1)
        in_image_upscaled = transform.rescale(original_greyscale, dilation, mode='constant',
                                              multichannel=False, anti_aliasing=True)
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(heatmap), percentile)
    abs_min = abs_max

    ax.imshow(heatmap, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    if overlay is not None:
        ax.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_original, alpha=alpha)
    return heatmap, overlay


def interpret_prediction(img_fpath, img_target, img_size, model, patchsize, figsize=(20, 20)):
    # load data
    image = np.array(transforms.Resize(img_size)(Image.open(img_fpath)))
    image = np.concatenate([np.expand_dims(np.array(image)[:,:,j], axis=0) for j in range(3)], axis=0)
    sample = image / 255.
    sample = np.expand_dims(sample, axis=0)
    heatmap = generate_heatmap_pytorch(model.to('cuda'), sample, img_target, patchsize, img_size)
    # plot
    fig = plt.figure(figsize=figsize)

    original_image = image.transpose([1,2,0])

    ax = plt.subplot(121)
    ax.set_title('original')
    plt.imshow(original_image / 255.)
    plt.axis('off')

    ax = plt.subplot(122)
    ax.set_title('heatmap')
    h, ov = plot_heatmap(heatmap, original_image, ax, dilation=0.5, percentile=99, alpha=.25)
    plt.axis('off')

    plt.show()
    return h, ov


if __name__ == '__main__':
    trained_model = train_basic_model(batch_size=512,
                                      learning_rate=2e-5,
                                      img_size=256)
    # torch.save(trained_model.to('cpu').state_dict(), "saved_model.pt")
    # Load data
    df_train, df_val, df_test = get_train_val_test_dfs(val_size=.3)
    y_train = df_train['target'].apply(int).values
    y_val = df_val['target'].apply(int).values
    # Predict
    df_train['pred'] = predict(trained_model, df_train)
    df_val['pred'] = predict(trained_model, df_val)
    df_test['pred'] = predict(trained_model, df_test)
    # Get aggregated accuracy
    train_acc = compute_aggregated_accuracy(df_train)
    val_acc = compute_aggregated_accuracy(df_val)
    print(f"train_acc={train_acc:.2%}\nval_acc={val_acc:.2%}")
