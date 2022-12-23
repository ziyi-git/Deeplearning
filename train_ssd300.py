


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 训练设备
    parser.add_argument('--device', default='cuda:0', help='device')
    # 检测的目标类别个数，不包括背景
    parser.add_argument('--num_classes', default=20, type=int, help='num_classes')
    # 训练数据集的根目录
    parser.add_argument('--data_path', default='./', hel)