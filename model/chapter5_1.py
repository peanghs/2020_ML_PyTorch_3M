import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

class ClassChapter5_1(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def starter(self, args):
        epochs = args.epochs
        batch_size = args.batch_size
        dropout_p = args.drop_out

        model = ClassChapter5_1_model(args).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)



    def data_loader(self,args):
        if __name__ == '__main__' :
            root_path = './data/'
        else:
            root_path = './model/data/'

        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(root=root_path, train= True, download=True,
                                  transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                                transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                                transforms.Normalize((0.1307,), (0.3081,))])),
             batch_size = args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(root=root_path, train=False,
                                  transform= transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),
                                                                 (0.3081,))])),
            batch_size=args.batch_size, shuffle=True)

class ClassChapter5_1_model(nn.Module):
    def __init__(self,args):
        super(ClassChapter5_1_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=2)
        self.conv2_drop = nn.Dropout2d(p=args.drop_out)
        print(self.conv2_drop.shape)
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
    def train(self, model, ): #  여기까지 하던 중