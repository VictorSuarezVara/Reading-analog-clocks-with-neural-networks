"""
            --------------
"""
'''#######################################################################################################'''
'''################################ ___ ResNet 50 pretrained MODEL ___ #############################'''
'''####################################################################################################################'''

'''IMPORTS'''
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import sys
from PIL import Image
from Aux_functions import pytorch_std_mean as auxFuncts
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

'''CUDA'''
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU")
else:
    device = torch.device("cpu")
    print("CPU")

'''#########################################'''
'''######### ___ Inicialitzation ___ ##############'''
'''######################################'''

'''HYPERPARAMETERS'''
input_model_size = 256
in_channel = 1
input_size = input_model_size*input_model_size*in_channel
batch_size = 256

dataSet_size = 50000
train_percentage = 0.8
test_percentage = 1-train_percentage

lr = 1e-3
mom = 0.9
lr_decay = True
threshold = 0.15

weight_decay = 0.0005

num_epochs = 40

train_size = m = int(dataSet_size * train_percentage)+1
test_size = int(dataSet_size * test_percentage)
if not train_size+test_size == dataSet_size:
    print("train_size+test_size is ", train_size+test_size, "and dataSet_size is ", dataSet_size)
    raise AssertionError()

    
#LOAD AND SAVE MODELS
load = False
save = False
where = 'resnet'
calcualte_norm = False

# HIPERPARAMETERS STANDARS
input_data_size = 512
num_classes_h = 12
num_classes_m = 1
mean_calculated = 0.4475
std_calculated = 0.2339
shuffle = True
save_ONNX = False
const = 29

# Val Loader
val_mean_calculated = 0.5310
val_std_calculated = 0.2998

# Config for wandb
config = dict(
    epochs=num_epochs,
    batch_size=batch_size,
    learning_rate=lr,
    momentum=mom,
    dataset="???",
    input_data_size=input_data_size,
    architecture="resnet50",
    mean_set=mean_calculated,
    std_set=std_calculated,
    weight_decay=weight_decay,
    lr_decay=lr_decay,
    threshold=threshold,

)

def make(config):
    # Make the data
    dataset = Clocks(
        csv_file="/home/vsuarez/Desktop/Neural Net Dataset/Datasetrgb/label.csv",
        root_dir="/home/vsuarez/Desktop/Neural Net Dataset/Datasetrgb/images",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(config.mean_set, config.mean_set, config.mean_set), std=(config.std_set, config.std_set, config.std_set)),

        ]),
    )

    dataset_Val = ClocksVal(
        csv_file="../0Datasets/Dataset_real/Datasetrgb/label.csv",
        root_dir="../0Datasets/Dataset_real/Datasetrgb/images",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(val_mean_calculated, val_mean_calculated, val_mean_calculated),
                                 std=(val_std_calculated, val_std_calculated, val_std_calculated)),


        ]),
    )

    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=test_set, batch_size=config.batch_size, shuffle=shuffle)

    val_set, aux = torch.utils.data.random_split(dataset_Val, [87, 0])
    val_loader = DataLoader(dataset=val_set, batch_size=config.batch_size, shuffle=False)

    # Little funtion to calculate the mean and standard desviation of the dataset
    if calcualte_norm:
        mean, std = auxFuncts.get_mean_std(val_loader)
        print(mean)
        print(std)
        sys.exit()



    # Model
    model = torchvision.models.resnet50(pretrained=True)
    freeze(model)
    unfreeze(model.layer4[0])
    unfreeze(model.layer4[1])
    unfreeze(model.layer4[2])
    model.avgpool = Identity()
    model.fc = Classifier(num_classes_h, num_classes_m)
    
    model = model.to(device=device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=threshold, verbose=True)

    # Criterion
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    return model, train_loader, test_loader, val_loader, criterion1, criterion2, optimizer, scheduler



'''#########################################'''
'''######### ___ DATASET ___ ##############'''
'''######################################'''
import os
import pandas as pd
from skimage import io
from torch.utils.data import (Dataset,DataLoader,)

class Clocks(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, str(index) + '.jpg')
        image = Image.open(img_path)

        y_label1 = int(self.annotations.iloc[index, 0])
        y_label2 = float(int(self.annotations.iloc[index, 1]) / 60)

        if self.transform:
            # newsize = (input_model_size, input_model_size)
            # image = image.resize(newsize)
            # image = image.convert('L')
            image = self.transform(image)

        return (image, y_label1, y_label2)

class ClocksVal(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, str(index) + '.jpg')
        image = Image.open(img_path)

        y_label1 = int(self.annotations.iloc[index, 0])
        y_label2 = float(int(self.annotations.iloc[index, 1]) / 60)

        if self.transform:
            newsize = (input_model_size, input_model_size)
            image = image.resize(newsize)
            # image = image.convert('L')
            image = self.transform(image)

        return (image, y_label1, y_label2)

'''#########################################'''
'''######### ___ ARCHITECTURE ___ ##############'''
'''######################################'''
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes_h=12, num_classes_m=1):
        super(Classifier, self).__init__()
        self.linearh1 = nn.Linear(in_features=131072, out_features=256)
        self.linearh2 = nn.Linear(in_features=256, out_features=num_classes_h)

        self.linearm1 = nn.Linear(in_features=131072, out_features=64)
        self.linearm2 = nn.Linear(in_features=64, out_features=num_classes_m)

        self.activationFullyLayerHour = nn.LeakyReLU(inplace=True)
        self.activationFullyLayerMinute = nn.LeakyReLU(inplace=True)
        self.finalActivationMinutes = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        hour = self.linearh1(x)
        hour = self.activationFullyLayerHour(hour)

        hour = self.linearh2(hour)

        minute = self.linearm1(x)
        minute = self.activationFullyLayerMinute(minute)

        minute = self.linearm2(minute)
        minute = self.finalActivationMinutes(minute)
        minute = minute.view(-1)

        return hour, minute
    
def init_weights(m):
    print(type(m))
    print(m)
    torch.nn.init.xavier_uniform(m.weight)
    m.bias.data.fill_(0.01)


def freeze(m):
    for p in m.parameters():
        p.requires_grad = False


def unfreeze(m):
    for p in m.parameters():
        p.requires_grad = True

''' Some aux functions'''
def check_metrics(loader, model, type_of_data):
    print("Metrics of ", type_of_data, " :")

    model.eval()

    num_correct_h = 0
    num_samples_h = 0

    num_samples_m = 0

    running_mae = 0

    with torch.no_grad():
        for data, targeth, targetm in loader:
            data = data.to(device=device)
            targeth = targeth.to(device=device)
            targetm = targetm.to(device=device)

            scores_h, scores_m = model(data)
            _, predictions_h = scores_h.max(1)

            num_correct_h += (predictions_h == targeth).sum()
            num_samples_h += predictions_h.size(0)

            error = torch.abs(scores_m - targetm).sum().data
            running_mae += error

            num_samples_m += scores_m.size(0)

        print("Got ", num_correct_h/num_samples_h, " for HOURS with accuracy: ", float(num_correct_h) / float(num_samples_h) * 100)
        print("For MINUTES we have mae: ", running_mae / num_samples_m * 100)
        

        if type_of_data == "train":
            wandb.log({"train_accuracy": float(num_correct_h) / float(num_samples_h), "train_mae": float(running_mae / num_samples_m * 100)})
        elif type_of_data == "test":
            wandb.log({"test_accuracy": float(num_correct_h) / float(num_samples_h), "test_mae": float(running_mae / num_samples_m * 100)})
        elif type_of_data == "val":
            wandb.log({"val_accuracy": float(num_correct_h) / float(num_samples_h), "val_mae": float(running_mae / num_samples_m * 100)})

    model.train()

def save_the_ONNX(loader, model):
    with torch.no_grad():
        for data, targeth, targetm in loader:

            data, targeth, targetm = data.to(device=device), targeth.to(device=device), targetm.float().to(device=device)

            # Save the model in the exchangeable ONNX format
            torch.onnx.export(model, data, "model.onnx")
            wandb.save("model.onnx")

            break

def load_checkpoint(model):
    print("=> Loading checkpoint")
    checkpoint = torch.load("Load and save models/" + where + "/my_checkpoint.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])

def save_checkpoint(state):
    print("=> Saving checkpoint")
    filename = "Load and save models/" + where + "/my_checkpoint.pth.tar"
    torch.save(state, filename)



'''#########################################'''
'''######### ___ TRAIN ___ ##############'''
'''######################################'''


def train(model, train_loader, test_loader, val_loader, criterion1, criterion2, optimizer, scheduler, config):
    # wandb ON
    wandb.watch(model, criterion1, log="all", log_freq=10)
    
    # Seters for wandb
    example_ct = 0  # number of examples seen
    
    for epoch in tqdm(range(config.epochs)):
        
        tic = time.time()
        
        cumulative_loss = 0
        for batch_idx, (data, targeth, targetm) in enumerate(train_loader):
            
            loss = train_batch(data, epoch, targeth, targetm, model, optimizer, criterion1, criterion2, scheduler)
            cumulative_loss += float(loss)
            example_ct += len(data)
            
            # update the bar + report metrics
            if (batch_idx + 1) % 25 == 0:
                train_log(loss, example_ct, epoch)
        
        if epoch > 15:
            scheduler.step(cumulative_loss / 78)
        
        wandb.log({"cumulative_loss": float(cumulative_loss / 78)})
        
        toc = time.time()
        print("La epoch numero ", epoch, " ha tardat: " + str(int(toc - tic) / 60) + " minuts")
        wandb.log({"epoch_time": (int(toc - tic) / 60)})
        
        if epoch == 19 or epoch == 29:
            check_metrics(train_loader, model, "train")
            check_metrics(test_loader, model, "test")
            check_metrics(val_loader, model, "val")


def train_batch(data, epoch, targeth, targetm, model, optimizer, criterion1, criterion2, scheduler):
    data, targeth, targetm = data.to(device=device), targeth.to(device=device), targetm.float().to(device=device)
    
    # forward
    out1, out2 = model(data)
    
    loss1 = criterion1(out1, targeth)
    loss2 = criterion2(out2, targetm) * const
    loss = loss1 + loss2
    
    losses_log(loss1, loss2, epoch)
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    
    # Step with optimizer
    optimizer.step()
    
    return loss


def train_log(loss, example_ct, epoch):
    loss = float(loss)
    
    # where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    # print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


def losses_log(loss1, loss2, epoch):
    # where the magic happens
    wandb.log({"epoch": epoch, "loss1": float(loss1), "loss2": float(loss2)})


''' _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_ '''
'''######### ___ MAIN ___ ##############'''
'''#####################################'''
def model_pipeline(hyperparameters):

    inicialtic = time.time()
    
    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, val_loader, criterion1, criterion2, optimizer, scheduler = make(config)
      if load:
          load_checkpoint(model)
          check_metrics(train_loader, model, "train")
          check_metrics(test_loader, model, "test")

      train(model, train_loader, test_loader, val_loader, criterion1, criterion2, optimizer, scheduler, config)

      if save:
          checkpoint = {"state_dict": model.state_dict()}
          save_checkpoint(checkpoint)

      check_metrics(train_loader, model, "train")
      check_metrics(test_loader, model, "test")
      check_metrics(val_loader, model, "val")

      if save_ONNX:
          save_the_ONNX(train_loader, model)

    toc = time.time()
    print("El programa ha tardat: " + str(int((toc - inicialtic) / 60)) + " minuts")


model_pipeline(config)

print("FINISH")