import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np
import json
import os
import time
import copy
import random
import torch.nn.functional as F
#

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        M1 = 16
        M2 = 32
        c = 1
        h1 = 14
        h2 = 14

        self.inception = Inception(2, M1, c)
        self.conv1 = BasicConv2d(3*c, M1, kernel_size=1, stride=1)
        self.conv2 = BasicConv2d(M1, M2, kernel_size=3, stride=1,padding=1)

        self.fc = nn.Sequential(
            
            nn.Linear(M2 * h1 * h2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x1, x2, x3):
        # MACCS
        H1 = 14
        W1 = 14
        # Pubchem
        H2 = 31
        W2 = 31
        # FP4
        H3 = 19
        W3 = 19

        x1 = x1.view(x1.size(0), 2, H1, W1)
        x2 = x2.view(x2.size(0), 2, H2, W2)
        x3 = x3.view(x3.size(0), 2, H3, W3)

        x = self.inception(x1, x2, x3)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        output = output.squeeze(-1)
        return output

class Inception(nn.Module):
    # self.inception = Inception(2, M1, M1)
    def __init__(self, in_channels, ch3x3, ch1): # ch3x3=16  ch1=1
        super(Inception, self).__init__()

        # 14*14
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3, kernel_size=3, stride = 1, padding = 1),
            # nn.MaxPool2d(kernel_size=2),
            BasicConv2d(ch3x3, ch1, kernel_size=3, stride=1, padding=1) # 14*14
        )

        # 31*31
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3, kernel_size=3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size=3, stride=2), # 15*15
            BasicConv2d(ch3x3, ch1, kernel_size=3, stride = 1, padding = 1), 
            nn.MaxPool2d(kernel_size=2, stride=1) # 14*14
        )

        # 19*19
        self.branch3 = nn.Sequential(
            # BasicConv2d(in_channels, ch3x3, kernel_size=(7,1), stride = (1,1)),
            BasicConv2d(in_channels, ch3x3, kernel_size=3, stride = 1, padding = 1), 
            nn.MaxPool2d(kernel_size=4, stride=1), ## 16*16
            BasicConv2d(ch3x3, ch1, kernel_size=3, stride = 1, padding = 1),   
            nn.MaxPool2d(kernel_size=3, stride=1)  # 14*14
        )
    def forward(self, x1, x2, x3):
    # def forward(self, x1, x2, x3, x4):
        branch1 = self.branch1(x1)
        branch2 = self.branch2(x2)
        branch3 = self.branch3(x3)

        outputs = [branch1, branch2, branch3]
        return torch.cat(outputs, 1)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

def process(phase,root,bit,whichkey):

    substrates_dict = json.load(open(root + 'Substrates_all' + str(bit) + '.json'))
    #k = 0
    imgs = []
    for key1, value1 in list(substrates_dict.items()):
        
        sub_channel = []
        pro_channel = []
        sub = open(root + phase + '/sub_channel'+str(whichkey)+'/rxn_sub' + key1 + '.txt', 'r', encoding='UTF-8') 
        pro = open(root + phase + '/pro_channel'+str(whichkey)+'/rxn_pro' + key1 + '.txt', 'r', encoding='UTF-8') 
        fh1 = sub.read()
        fh2 = pro.read()
        fh1_5 = fh1.split('\n')
        fh2_5 = fh2.split('\n')
        fh1_5.pop()
        fh2_5.pop()
        for i in range(len(fh1_5)):  # 4
            words = fh1_5[i].split()
            for j in range(len(words)):
                sub_channel.append(float(words[j]))  # 168
        for m in range(len(fh2_5)):  # 4
            lines = fh2_5[m].split()
            for n in range(len(lines)):
                pro_channel.append(float(lines[n]))  # 168

        img = [sub_channel, pro_channel]
        imgs.append(img)
        # np.save(root + phase + '/two_channel/rxn' + str(k) + '.npy', img1)
        sub.close()
        pro.close()

    return imgs

# data = MyDataset(root, b_all, 0, 'data', transform=transforms.ToTensor())
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, b, bit, phase, transform=None):
        self.imgs0 = process(phase, root, bit, 0)
        self.imgs1 = process(phase, root, bit, 1)
        self.imgs2 = process(phase, root, bit, 2)
        # self.imgs3 = process(phase, root, bit, 5)

        self.b = b
        self.transform = transform

    def __getitem__(self, index):
        x0 = self.imgs0[index]
        x1 = self.imgs1[index]
        x2 = self.imgs2[index]
        # x3 = self.imgs3[index]
        data = [x0, x1, x2]
        # data = [x0, x1, x2, x3]
        label = self.b[index]
        if self.transform is not None:
            for i in range(len(data)):
                 data[i] = torch.Tensor(data[i])

        return data, label

    def __len__(self):
        return len(self.imgs0)

def trainAndval():

    cnn = CNN()
    cnn = nn.DataParallel(cnn)
    cnn.cuda()
    summary(cnn, input_size=[(1, 2, 196),(1, 2, 961),(1, 2, 361)], batch_size=1)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR,weight_decay=1e-5)
    loss_func = nn.MSELoss() 
    mae_func = nn.L1Loss()  
    scheduler = MultiStepLR(optimizer, milestones=[500,1000,1800], gamma=0.1)
    best_test_mae = float('inf')  # Initialize with a large value
    best_model_weights = None

    train_MSElosses = []
    train_MAElosses = []

    val_MSElosses = []
    val_MAElosses = []
    val_num = len(val_loader.dataset)
    for epoch in range(EPOCH):
        cnn.train()
        outputs = []
        labels = []
        time_start = time.time()
        running_loss = 0.0 


        for step, (x1, y1) in enumerate(train_loader):
            b_x0 = x1[0].cuda()
            b_x1 = x1[1].cuda()
            b_x2 = x1[2].cuda()
            b_y1 = y1.cuda()
            b_y1 = b_y1.float()
            output = cnn(b_x0, b_x1, b_x2).float()

            loss = loss_func(output, b_y1)
            train_mae = mae_func(output, b_y1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() 

            outputs += output.tolist()
            labels += b_y1.tolist()
        result_txt = torch.tensor([outputs,labels])
        result_txt = torch.transpose(result_txt,0,1)
        result_txt = torch.round(result_txt,decimals=3)

        np.savetxt(result_file+"/train_result.txt",np.array(result_txt))
        time_end = time.time()

        train_mse = running_loss / len(train_loader)
        train_MSElosses.append(float(train_mse))
        train_MAElosses.append(float(train_mae))


        scheduler.step()

        val_act = np.zeros((val_num, 1))
        val_eva = torch.zeros(val_num, 1).cuda()
        error = .0
        error_square = .0

        if (epoch%50==0):
            cnn.eval()
            with torch.no_grad():
                for step, (test_x1, test_y1) in enumerate(val_loader):
                    x0 = test_x1[0].cuda()
                    x1 = test_x1[1].cuda()
                    x2 = test_x1[2].cuda()
                    output1 = cnn(x0,x1,x2) 

                    val_eva[step] = output1
                    val_act[step] = test_y1

                val_eva = val_eva.cpu().numpy()
                for i in range(val_num):
                    error += abs(val_act[i] - val_eva[i])
                    error_square +=  (val_act[i]-val_eva[i])**2
                mae = error / val_num
                mse = error_square / val_num

                val_MAElosses.append(float(mae))
                val_MSElosses.append(float(mse))

                if mae < best_test_mae:
                    best_test_mae = mae
                    best_model_weights = copy.deepcopy(cnn.state_dict())
                    
                    best_val_eva = val_eva.squeeze().tolist()
                    bets_val_act = val_act.squeeze().tolist()

        if (epoch % 50 == 0):
            print(
                'Train Epoch: {} \tTime: {} \ttrain_mse: {:.6f} \ttrain_mae: {:.6f} \tLr:{:.6f} \tval_mae:{:.4f} \t'.format(
                    epoch, time_end - time_start, train_mse, train_mae.data.cpu().numpy(),
                    optimizer.state_dict()['param_groups'][0]['lr'],
                    mae[0]))


    for elem in best_val_eva:
        val_outputs.append(elem)
    for elem in bets_val_act:
        val_labels.append(elem)

    if best_model_weights is not None:
        torch.save(best_model_weights, result_file+"/best_model_weights.pth")
    np.savetxt(result_file+"/train_mse_loss.txt", np.array(train_MSElosses))
    np.savetxt(result_file+"/train_mae_loss.txt", np.array(train_MAElosses))
    np.savetxt(result_file+"/val_mse_loss.txt", np.array(val_MSElosses))
    np.savetxt(result_file+"/val_mae_loss.txt", np.array(val_MAElosses))
    return mae,mse

def test():

    model = CNN().cuda()
    checkpoint = torch.load(result_file+'/best_model_weights.pth')
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    model.load_state_dict(new_state_dict)

    model.eval()

    test_num = len(test_loader.dataset)
    test_act = np.zeros((test_num, 1))
    test_eva = torch.zeros(test_num, 1).cuda()
    error = .0
    error_square = .0

    print("Test dataset size:", test_num)
    with torch.no_grad():
        for step, (text_x1, text_y1) in enumerate(test_loader):
            x0 = text_x1[0].cuda()
            x1 = text_x1[1].cuda()
            x2 = text_x1[2].cuda()
            output1 = model(x0, x1, x2)
            test_eva[step] = output1
            test_act[step] = text_y1

        test_eva = test_eva.cpu().numpy()
        for i in range(test_num):
            error += abs(test_act[i] - test_eva[i])
            error_square += (test_act[i] - test_eva[i]) ** 2
        mae = error / test_num
        mse = error_square / test_num
        test_eva = test_eva.squeeze().tolist()
        test_act = test_act.squeeze().tolist()

    for elem in test_eva:
        test_outputs.append(elem)
    for elem in test_act:
        test_labels.append(elem)
    return mae,mse

start_time = time.time()

EPOCH = 2000
BATCH_SIZE = 64
LR = 0.001
mae_sum = .0
last_mae_sum = .0

net_seed = 2341
cv_ceed = 2341
random.seed(net_seed)
np.random.seed(net_seed)
torch.manual_seed(net_seed)
torch.cuda.manual_seed(net_seed)
os.environ['PYTHONHASHSEED'] = str(net_seed)
torch.cuda.manual_seed_all(net_seed)
cudnn.benchmark = False
cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

root = "../test1/"
result_file = "./result/main5-1118"
# training set
b_all = json.load(open("../test1/data/b_all.json"))

data = MyDataset(root, b_all, 0, 'data', transform=transforms.ToTensor())

train_size = int(0.6*len(data))
val_size = int(0.2*len(data))
test_size = len(data)-train_size-val_size
train_set, val_set, test_set = torch.utils.data.random_split(data, [train_size, val_size,test_size])

print("train_set:",len(train_set))
print("val_set:",len(val_set)) # 8401
print("test_set:",len(test_set)) # 8402
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


val_outputs = []
val_labels = []
val_mae,val_mse = trainAndval()
val_txt = torch.tensor([val_outputs, val_labels])
val_txt = torch.transpose(val_txt, 0, 1)
val_txt = torch.round(val_txt, decimals=3)
np.savetxt(result_file+"/val_result.txt", np.array(val_txt))
print("The MAE on the validation dataset",val_mae)


test_outputs = []
test_labels = []
test_mae,test_mse = test()
print("The MAE on the test dataset",test_mae)
print("The MSE on the test dataset",test_mse)
test_txt = torch.tensor([test_outputs, test_labels])
test_txt = torch.transpose(test_txt, 0, 1)
test_txt = torch.round(test_txt, decimals=3)
np.savetxt(result_file+"/test_result.txt", np.array(test_txt))


end_time = time.time()

execution_time = end_time - start_time
print(f"The execution time of the code{execution_time} s")
print(f"The execution time of the code{execution_time/3600} h")
