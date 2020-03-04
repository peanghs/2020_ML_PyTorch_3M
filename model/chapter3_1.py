import torch


class ClassChapter3_1(object):
    def tensor_config(self):
        print('--기본 작업--')
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        print(x)
        print("size : ", x.size())
        shape = x.shape
        rank = x.ndimension()
        print(f"shape : {0}".format(shape))
        print(f"rank : {rank}")

        print('--랭크 늘리기--')
        x = torch.unsqueeze(x, 0)
        print(x)
        rank = x.ndimension()
        print(f"shape : {0}".format(x.shape))
        print(f"rank : {rank}")

        print('--랭크 줄이기--')
        x = torch.squeeze(x)
        print(x)
        rank = x.ndimension()
        print(f"shape : {0}".format(x.shape))
        print(f"rank : {rank}")

        print('--추가 : 텐서 붙이기--')
        y = torch.tensor([[10, 11, 12]])
        x = torch.cat([x, y], dim=0)
        print(x)

        print('--view 사용--')
        x = x.view(12)
        print(x)
        print("Size :", x.size())
        print("Shape :", x.shape)
        print("rank :", x.ndimension())

        print('--에러 회피--')
        try:
            x = x.view(2, 8)
            print(x)
        except Exception as e:
            print(e)

    def tensor_multiple(self):
        print('--텐서 곱하기--')
        w = torch.randn(5, 3, dtype=torch.float)
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        print(f'w size: {w.size()}')
        print(f'x size: {x.size()}')
        print(f'w:{w}')
        print(f'x:{x}')

        b = torch.randn(5, 2, dtype=torch.float)
        print(f'b:{b.size()}')
        print(f'b:{b}')

        wx = torch.mm(w, x)
        print(f'wx size:{wx.size()}')
        print(f'wx:{wx}')

        result = wx + b
        print(f'result size:{result.size()}')
        print(f'result:{result}')

    def autograd_train(self):
        w = torch.tensor(1.0, requires_grad=True)  # True 시 w.grad에 기울기 저장
        a = w*3
        l = w**2
        l.backward()
        print(f'l을 w로 미분한 값은 {w.grad}')