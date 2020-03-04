import torch
import pickle
import matplotlib.pyplot as plt


class ClassChapter3_2(object):
    def image_recovery(self):
        if __name__ == '__main__':
            broken_image = torch.FloatTensor(pickle.load(open('./image/broken_image_t.p', 'rb'), encoding='latin1'))
        else:
            broken_image = torch.FloatTensor(pickle.load(open('./model/image/broken_image_t.p', 'rb'), encoding='latin1'))

        plt.imshow(broken_image.view(100, 100))
        # plt.show()
        return broken_image

    def weird_function(self, x, n_iter=5):
        h = x
        filt = torch.tensor([-1. / 3, 1. / 3, -1. / 3]).cuda()
        for i in range(n_iter):
            zero_tensor = torch.tensor([1.0 * 0]).cuda()
            h_l = torch.cat((zero_tensor, h[:-1]), 0)
            h_r = torch.cat((h[1:], zero_tensor), 0)
            h = filt[0] * h + filt[2] * h_l + filt[1] * h_r
            if i % 2 == 0:
                h = torch.cat((h[h.shape[0] // 2:], h[:h.shape[0] // 2]), 0)
        return h

    def distance_loss(self, hypothesis, broken_image):
        return torch.dist(hypothesis, broken_image)

    def distance_calc(self,args):
        random_tensor = torch.randn(10000, dtype=torch.float).cuda()
        lr = args.lr
        epochs = args.epochs
        for i in range(epochs):
            broken_image = self.image_recovery()
            random_tensor.requires_grad_(True)
            hypothesis = self.weird_function(x=random_tensor)
            loss = self.distance_loss(hypothesis.cuda(), broken_image.cuda()) # cuda 로 계산 시도. 딱히 시간이 빨라진 것 같진 않음..
            loss.backward()
            with torch.no_grad():
                random_tensor = random_tensor - lr*random_tensor.grad
            if i % 1000 == 0:
                print(f'Loss at {i} = {loss.item()}')
        return random_tensor
