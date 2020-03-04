import torch
import matplotlib.pyplot as plt
from model import chapter2_1, chapter2_2, chapter2_3

class Starter(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'현재 모드 : {self.device}')

    def load_model(self, args):
        mode = args.mode
        print(f'----{mode}를 시작합니다----')
        if mode == 'chapter_2_1':
            play_chapter_2 = chapter2_1.ClassChapter2_1()
            print(play_chapter_2.tensor_config())
            print(play_chapter_2.tensor_multiple())
            print(play_chapter_2.autograd_train())
        if mode == 'chapter_2_2':
            play_chapter_2 = chapter2_2.ClassChapter2_2()
            random_tensor = play_chapter_2.distance_calc()
            random_tensor = random_tensor.cpu()
            plt.imshow(random_tensor.view(100, 100).data)
            plt.show()
        if mode == 'chapter_2_3':
            ClassProcess = chapter2_3.ClassChapter2_3_eval()
            ClassProcess.process(args)











