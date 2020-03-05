from torchvision import datasets, transforms, utils
from torch.utils import data

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import os


class ClassChapter4_1(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def starter(self, args):
        train_loader, test_loader = self.data_load(args)
        print('-----데이터 살펴보기-----')
        self.glance_data(train_loader)
        print('-----학습 및 평가-----')
        dropout_p = args.drop_out
        model = ClassChapter4_1_model(dropout_p).to(self.device)
        optimizer = opt.SGD(model.parameters(), lr=args.lr)
        epochs = args.epochs
        process = ClassChapter4_1_process()

        if __name__ == '__main__':
            backup_path = './backup/'
        else:
            backup_path = './model/backup/'
        backup_file = f'{backup_path}c4_1_model.pt'
        if os.path.isfile(backup_file):
            model.load_state_dict(torch.load(backup_file))
            print(f'모델 로드 경로 :{backup_file}')
            print('모델 로드 완료')
        else:
            print('신규 학습 시작')
        try:
            if not os.path.isdir(backup_path):
                os.makedirs(backup_path)
                print(f'백업 경로 생성 : {backup_path}')
        except OSError:
            print(f'백업 경로 생성 실패 : {backup_path}')

        for epoch in range(1, epochs+1):
            process.train(model, train_loader, optimizer, self.device)
            test_loss, test_accuracy = process.evaluate(model, test_loader, self.device)
            if epoch % 10 == 0:
                print(f'[{epoch}] Test Loss:{test_loss:.4f}, Accuracy:{test_accuracy:.2f}%')

        torch.save(model.state_dict(), backup_file)
        print(f'모델 백업 경로 :{backup_file}')

    def data_load(self, args):
        if __name__ == '__main__':
            root_path = './data/'
        else:
            root_path = './model/data/'
        trainset = datasets.FashionMNIST(root=root_path, train=True, download=True, transform=transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        testset = datasets.FashionMNIST(root=root_path, train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        batch_size = args.batch_size
        train_loader = data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
        test_loader = data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader

    def glance_data(self, train_loader):
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        img = utils.make_grid(images, padding=0)  # make_grid 여러개 이미지 함께 보기(배치 수)
        np_img = img.numpy()  # plt 호환을 위해서는 텐서 -> numpy로 바꿔야함
        plt.figure(figsize=(10, 7))
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.show()
        classes = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal',
                   6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
        for label in labels:
            index = label.item()
            # print(classes[index])

        idx = 1
        item_img = images[idx]
        item_np_img = item_img.squeeze().numpy()
        plt.title(classes[labels[idx].item()])
        # print(item_np_img.shape)
        plt.imshow(item_np_img, cmap='gray')
        plt.show()

class ClassChapter4_1_model(nn.Module):
    def __init__(self, dropout_p):
        super(ClassChapter4_1_model, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout_p = dropout_p

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout_p)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training, p=self.dropout_p)
        x = self.fc3(x)
        return x

class ClassChapter4_1_process(object):
    def train(self, model, train_loader, optimizer, device):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

    def evaluate(self, model, test_loader, device):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100 * correct / len(test_loader.dataset)
        return test_loss, test_accuracy
