import argparse
from solver import Starter


def main(args):
    starter = Starter(args)
    if args.mode == 'chapter_9_1':  # 다른 조건이 필요할 때 사용
        pass
    else:
        starter.load_model(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ---- 설정 ----
    parser.add_argument('--mode', type=str, default='chapter_3_2',
                        choices=['chapter_3_1', 'chapter_3_2', 'chapter_3_3'])
    parser.add_argument('--lr', type=float, default='0.01')
    parser.add_argument('--epochs', type=int, default='5000')

    args = parser.parse_args()
    print(args)
    main(args)
