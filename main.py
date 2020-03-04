import argparse
from solver import Starter

def main(args):
    starter = Starter(args)
    if args.mode == 'chapter_2_1' or 'chapter_2_2':
        starter.load_model(args)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # ---- 설정 ----
    parser.add_argument('--mode', type=str, default='chapter_2_3', choices=['chapter_2_1','chapter2_2','chapter2_3'])
    parser.add_argument('--lr', type=float, default='0.01')
    parser.add_argument('--epochs', type=int, default='5000')

    args = parser.parse_args()
    print(args)
    main(args)
