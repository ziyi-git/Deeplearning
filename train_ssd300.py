
def main(parser_data):
    


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 训练设备
    parser.add_argument('--device', default='cuda:0', help='device')
    # 检测的目标类别个数，不包括背景
    parser.add_argument('--num_classes', default=20, type=int, help='num_classes')
    # 训练数据集的根目录
    parser.add_argument('--data_path', default='./', help='data_path')
    # 文件保存地址
    parser.add_argument('--output_dir', default='./', help='path to save')
    # 若需要接着上次训练，则指定上次训练的权重保存地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=15, type=int, metavar='N', help='number of total epochs')
    # 训练的batch size
    parser.add_argument('--batch_size', default=4, type=int, metavat='N', help='batch size')

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    main(args)

    