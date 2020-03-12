import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import os

class ClassChapter5_1(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def starter(self, args):
        epochs = args.epochs
        batch_size = args.batch_size
        dropout_p = args.drop_out

        model = ClassChapter5_1_model(args).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        train_loader, test_loader = self.data_loader(args)
        process = ClassChapter5_1_process()

        if __name__ == '__main__':
            backup_path = './backup/'
        else:
            backup_path = './model/backup/'
        backup_file = f'{backup_path}c5_1_model.pt'
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

        for epoch in range(1, epochs + 1):
            process.train(model, train_loader, optimizer, epoch, self.device)
            test_loss, test_accuracy = process.evaluate(model, test_loader, self.device)

            if epoch % 10 == 0:
                print(f'[{epoch}] Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

        torch.save(model.state_dict(), backup_file)
        print(f'모델 백업 경로 :{backup_file}')

    def data_loader(self,args):
        if __name__ == '__main__' :
            root_path = './data/'
        else:
            root_path = './model/data/'

        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(root=root_path, train= True, download=True,
                                  transform=transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize((0.1307,), (0.3081,))])),
             batch_size = args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(root=root_path, train=False,
                                  transform= transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),
                                                                 (0.3081,))])),
            batch_size=args.batch_size, shuffle=True)
        return train_loader, test_loader

class ClassChapter5_1_model(nn.Module):
    def __init__(self,args):
        super(ClassChapter5_1_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=args.drop_out)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ClassChapter5_1_process(object):
    def train(self, model, train_loader, optimizer, epoch, device):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # if batch_idx % 200 == 0 :
            #     print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
            #           f' ({100.*batch_idx/len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    def evaluate(self,model,test_loader, device):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader :
                data, target = data.to(device), target.to(device)
                output = model(data)

                test_loss += F.cross_entropy(output, target, reduction='sum').item()

                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100 * correct/len(test_loader.dataset)
        return test_loss, test_accuracy

