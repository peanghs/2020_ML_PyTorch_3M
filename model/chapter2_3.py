import torch
import numpy
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F


class ClassChapter2_3_prepare(object):
    def prepare_nn(self):
        n_dim = 2
        x_train, y_train = make_blobs(n_samples=80, n_features=n_dim, centers=[[1,1],[-1,-1],[1,-1],[-1,1]], shuffle=True, cluster_std=0.3)
        x_test, y_test = make_blobs(n_samples=20, n_features=n_dim, centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]],
                                    shuffle=True, cluster_std=0.3)
        y_train = self.label_map(y_train, [0,1], 0)
        y_train = self.label_map(y_train, [2,3], 1)
        y_test = self.label_map(y_test, [0,1], 0)
        y_test = self.label_map(y_test, [2,3], 1)

        plt.figure()
        self.vis_data(x_train, y_train, c='r')
        plt.show()
        return x_train, x_test, y_train, y_test

    def label_map(self, y_, from_, to_):
        y= numpy.copy(y_)
        for f in from_:
            y[y_==f]=to_
        return y

    def vis_data(self, x, y=None, c='r'):
        if y is None:
            y = [None] * len(x)
        for x_, y_ in zip(x, y):
            if y_ is None:
                plt.plot(x_[0], x_[1], '*', markerfacecolor='none', markeredgecolor=c)
            else:
                plt.plot(x_[0], x_[1], c + 'o' if y_ == 0 else c + '+')

class ClassChapter2_3_train(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ClassChapter2_3_train, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_tensor):
        linear1 = self.linear_1(input_tensor)
        relu = self.relu(linear1)
        linear2 = self.linear_2(relu)
        output = self.sigmoid(linear2)
        return output

class ClassChapter2_3_eval(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def process(self, args):
        prepare = ClassChapter2_3_prepare()
        x_train, x_test, y_train, y_test = prepare.prepare_nn()
        learning_rate = args.lr
        epochs = args.epochs

        x_train = torch.FloatTensor(x_train).to(self.device)
        x_test = torch.FloatTensor(x_test).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)

        model = ClassChapter2_3_train(2, 5).to(self.device)  # 변수를 cuda 로 준 경우 모델도 device 설정을 해줘야 함
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        model.eval()
        test_loss_before = criterion(model(x_test).squeeze(), y_test)
        print(f'트레이닝 전 테스트 Loss : {test_loss_before.item()}')

        if __name__ == '__main__':
            backup_path = './backup/'
        else:
            backup_path = './model/backup/'
        backup_file = f'{backup_path}model.pt'
        if os.path.isfile(backup_file):
            model.load_state_dict(torch.load(backup_file))
            print(f'모델 로드 경로 :{backup_file}')
            print('모델 로드 완료')
        try:
            if not os.path.isdir(backup_path):
                os.makedirs(backup_path)
                print(f'백업 경로 생성 : {backup_path}')
        except OSError:
            print(f'백업 경로 생성 실패 : {backup_path}')

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            train_output = model(x_train)
            train_loss = criterion(train_output.squeeze(), y_train)
            if epoch % 1000 == 0:
                print(f"{epoch}의 Training Loss : {train_loss.item()}")
            train_loss.backward()
            optimizer.step()

        model.eval()
        test_loss = criterion(model(x_test).squeeze(), y_test)
        print(f'트레이닝 후 테스트 로스 : {test_loss.item()}')
        print(f'트레이닝 전후 로스 차이 : {test_loss_before.item() - test_loss.item()} 감소')

        torch.save(model.state_dict(), backup_file)
        print(f'모델 백업 경로 :{backup_file}')


