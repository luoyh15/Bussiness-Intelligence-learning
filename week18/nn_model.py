import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, TensorDataset
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

# use gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# 网络定义
class Net(nn.Module):
    def __init__(self, n_num, num_embeddings, embedding_dim):
        super(Net, self).__init__()
        self.embeddings = nn.ModuleList()
        for n_embd, embd_d in zip(num_embeddings, embedding_dim):
            self.embeddings.append(nn.Embedding(n_embd, embd_d))
        n_features = n_num + sum(embedding_dim)
        n_base = 64
        self.fc1 = nn.Linear(n_features, n_base*8)
        self.fc2 = nn.Linear(n_base*8, n_base*4)
        self.fc3 = nn.Linear(n_base*4, n_base*2)
        self.fc4 = nn.Linear(n_base*2, n_base)
        self.fc5 = nn.Linear(n_base, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.active = nn.ReLU()
    
    def forward(self, x_num, x_cat):
        # print(x_num.shape, x_cat.shape)
        x = x_num
        for i, embd in enumerate(self.embeddings):
            # print(x, x_cat, x_cat[:, i])
            x = torch.cat((x, embd(x_cat[:, i])), dim=1)

        x = self.active(self.fc1(x)) 
        x = self.active(self.fc2(x))
        # x = self.dropout(x)
        x = self.active(self.fc3(x))
        x = self.active(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = self.dropout(x)
        x = self.fc5(x)
        return torch.squeeze(x, 1)

    # 重置网络参数（Kflod需要）
    def reset_parameter(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


# 自定义L2正则，pytorch中的L2正则是靠Adam的weight_decay实现的，和通常的L2正则有区别，所以这里自定义了。（后来发现AdamW似乎可以和通常的L2正则对应）
class Regularization(nn.Module):
    def __init__(self, alpha):
        super(Regularization, self).__init__()
        self.alpha = alpha
        # self.alpha_embd = aplha_embd
    
    def forward(self, model):
        loss, loss_embd = 0, 0
        for name, param in model.named_parameters():
            # print(name)
            if 'weight' in name:
                if 'embedding' in name:
                    loss_embd += torch.norm(param, p=2)
                else:
                    loss += torch.norm(param, p=2)
        return self.alpha*loss


# train for one epoch
def train_step(dataset, model, optimizer, criterion, regularization, batch_size=512):
    # put model into device
    model.to(device)
    # construct dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size)
    # start train
    total_loss = 0.0
    for data in dataloader:
        # get the inputs
        X_num, X_cat, y = data[0].to(device), data[1].to(device), data[2].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward+backward+optimize
        outputs = model(X_num, X_cat)
        loss = criterion(outputs, y)
        total_loss += loss.item()
        loss += regularization(model)
        loss.backward()
        optimizer.step()       
    
    return total_loss/len(dataset)
    
# evaluation the model on dataset
def eval(dataset, model, criterion, batch_size=512):
    # print(f'Using {device} device')
    # put model into device
    model.to(device)
    # construct dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size)
    # start evaluate
    total_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            # get the inputs
            X_num, X_cat, y = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs = model(X_num, X_cat)
            total_loss += criterion(outputs, y)
    return total_loss/len(dataset)


def train(train_set, test_set, model, epochs=1000):
    # loss function
    criterion = nn.L1Loss(reduction='sum')
    # l2 regularization
    regularization = Regularization(0.1)
    
    # min_loss for save the best model
    min_loss = float('inf')
    lr = 0.001 # 学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        # 动态调整学习率
        if epoch in [400, 700, 900]:
            lr *= 0.1
            model.load_state_dict(best_model)
            for g in optimizer.param_groups:
                g['lr'] = lr
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            print(f'{epoch}: load best model')
        # train one epoch
        train_loss = train_step(train_set, model=model, optimizer=optimizer, criterion=criterion, regularization=regularization)
        # print statistics
        test_loss = eval(test_set, model=model, criterion=criterion)
        print(f'[{epoch+1}], train loss:{train_loss:.3f}, test loss:{test_loss:.3f}')
        # save the best model's parameters
        if test_loss < min_loss:
            best_model = model.state_dict()
            # print(best_model)
            min_loss = test_loss
            print('best model found.')

    return best_model


if __name__ == '__main__':
    # 数据加载
    path = 'data/nn/'
    train_X = pd.read_pickle(path+'train_X.pkl')
    test_X = pd.read_pickle(path+'test_X.pkl')
    train_y = pd.read_pickle(path+'train_y.pkl')

    categorical_cols = ['model', 'brand', 'bodyType', 'fuelType', 'notRepairedDamage', 'city']
    numerical_cols = [col for col in train_X.columns if col not in categorical_cols]
    print(f'number of numerical features:{len(numerical_cols)}')
    categorical_cols = ['brand', 'bodyType', 'fuelType', 'notRepairedDamage', 'city']
    # 分开处理数值特征和类别特征
    train_X_num, test_X_num = train_X[numerical_cols], test_X[numerical_cols]
    train_X_cat, test_X_cat = train_X[categorical_cols], test_X[categorical_cols]
    # vocaborary size
    num_embeddings = list(pd.concat([train_X_cat, test_X_cat], axis=0).nunique())
    print(num_embeddings)
    # embedding dimmension
    embedding_dim = [1, 1, 1, 1, 1]
    print(f'number of embedding dims:{sum(embedding_dim)}')

    # model initialize
    n_num = len(numerical_cols)
    model = Net(n_num, num_embeddings, embedding_dim).to(device)
    print(list(model.children()))
    
    # 10折交叉验证
    kf = KFold(n_splits=5, shuffle=True)
    results = [] #保存预测结果
    X_num, X_cat, y = train_X_num.values, train_X_cat.values, train_y.values
    for i, (train_index, test_index) in enumerate(kf.split(y)):
        print(f'{i+1} fold:')
        X_num_train, X_num_test = X_num[train_index], X_num[test_index]
        X_cat_train, X_cat_test = X_cat[train_index], X_cat[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_set = TensorDataset(torch.Tensor(X_num_train), torch.LongTensor(X_cat_train), torch.Tensor(y_train))
        test_set = TensorDataset(torch.Tensor(X_num_test), torch.LongTensor(X_cat_test), torch.Tensor(y_test))  
        # start training
        best_model = train(train_set, test_set, model, epochs=1000)
        # make prediction
        model.load_state_dict(best_model)
        # 预测
        with torch.no_grad():
            test_num = torch.Tensor(test_X_num.values).to(device)
            test_cat = torch.LongTensor(test_X_cat.values).to(device)
            results.append(model(test_num, test_cat).cpu())

        for name, param in model.named_parameters():
            if 'embedding' in name:
                print(name, param)
        # reset
        model.reset_parameter()

    submission = pd.read_csv('data/used_car_sample_submit.csv')
    submission['price'] = torch.stack(results).mean(dim=0)
    submission.to_csv('nn_submission.csv')
