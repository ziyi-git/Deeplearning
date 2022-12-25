import os
import datetime

import torch

import transforms
from my_dataset import VOCDataSet


def create_model(num_classes=21):
    backbone = Backbone()  # 特征提取器
    model = SSD300(backbone=backbone, num_classes=num_classes)

    pre_ssd_path = '***'
    if os.path.exists(pre_ssd_path) is False:
        raise FileNotFoundError('*** not found in {}'.format(pre_ssd_path))
    pre_model_dict = torch.load(pre_ssd_path, map_location='cpu')
    pre_weights_dict = pre_model_dict['model']

    # 不加载类别预测器的权重，因为是voc和coco不同，但是可以使用回归预测器的权重
    del_conf_loc_dict = {}
    for k, v in pre_weights_dict.items():
        split_key = k.split('.')
        if 'conf' in k:
            continue
        del_conf_loc_dict.update({k: v})
    
    missing_keys, unexpected_keys = model.load_state_dict(del_conf_loc_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    return model    


def main(parser_data):
    # 指定GPU或CPU，Q：如何进行多GPU训练
    device = torch.device(parser_data.device if torch.cuda.is_available() else 'cpu')
    print('Using device {} training.'.format(device.type))

    if not os.path.exists('save_weights'):
        os.makedirs('save_weights')
    
    results_file = 'results{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    data_transform = {
        'train': transforms.Compose([
            transforms.SSDCropping(),  # Q: 裁剪应该旨在增加训练样本，裁剪后GT也要改变？
            transforms.Resize(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlips(),
            transforms.Normalization(),
            # ???：输出default box集合的正样本和负样本(default_box的位置固定，正样本和GT匹配最佳且满足IOU>0.5)
            transforms.AssignGTtoDefaultBox()
        ]),
        'val': transforms.Compose([
            transforms.Resize(),
            transforms.ToTensor(),
            transforms.Normalization()
        ])
    }

    VOC_root = parser_data.data_path
    if os.path.exists(os.path.join(VOC_root, 'VOCdevkit')) is False:
        raise FileNotFoundError('VOCdevkit does not in path: {}'.format(VOC_root))
    
    # VOCdevkit/VOC2012/ImageSets/Main/train.txt
    train_dataset = VOCDataset(VOC_root, '2012', data_transform['train'], train_set='train.txt')
    # 训练时batch size必须大于1.
    batch_size = parser_data.batch_size
    assert batch_size > 1, 'batch size must be greater than 1'  # Q：assert条件不满足，是否就推出程序？
    # 防止最后一个batch_size=1，如果是就舍去
    drop_last = True if len(train_dataset) % batch_size == 1 else False
    nw = min([os.cput_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn,  # 核对函数
        drop_last=drop_last
    )

    # VOCdevkit/VOC2012/ImageSets/Main/val.txt
    val_dataset = VOCDataset(VOC_root, '2007', data_transform['val'], train_set='val.txt')
    val_data_loader = torch.utils.data.Dataloader(
        val_dataset,
        batch_size=batch_size,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn  # ?：为何不是val_dataset.collate_fn
    )

    model = create_model(num_classes=parser_data.num_classes+1)
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]  # Q：可以更新的参数部分？
    optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)

    # learning date scheduler，Q：step_size指每个epoch增大5倍？
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    # 如果指定了上次训练保存的权重文件地址，则续接上次结果训练
    if parser_data.resume != '':
        check_point = torch.load(parser_data.resume, map_location='cpu')  # Q：在cpu上加载保存于GPU的模型？
        model.load_state_dict(check_point['model'])
        optimizer.load_state_dict(check_point['optimizer'])
        lr_scheduler.load_state_dict(check_point['lr_scheduler'])
        parser_data.start_epoch = check_point['epoch'] + 1
        print('the training process from epoch {}...'.format(parser_data.start_epoch))
    
    train_loss = []
    learning_rate = []
    val_map = []

    # 提前加载验证集数据，以免每次验证都重新加载一次
    val_data = get_coco_api_from_dataset(val_data_loader.dataset)  # Q：这里是说一次性加载所有的数据？
    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        mean_loss, lr = utils.train_one_epoch(model=model,optimizer=optimizer, data_loader=train_data_loader, device=device, epoch=epoch, print_freq=50)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update learning rate
        lr_scheduler.step()

        # Q：是否使用data_set参数后就是一直把数据保存在GPU上，从而忽略了data_loader这种以batch加载数据的方式
        coco_info = utils.evaluate(model=model, data_loader=val_data_loader, device=device, data_set=val_data)
    
        # write info txt
        with open(results_file, 'a') as f:
            result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")
    
        val_map.append(coco_info[1])

        # save weights
        save_files = {
            'model': model.state_dict(),  # Q：state_dict()是什么意思？
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch
        }

        torch.save(save_files, './save_weights/ssd300-{}.pth'.format(epoch))
    
    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, lr)
    
    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


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

    