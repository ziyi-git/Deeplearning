import math
import sys
import time

import torch

from train_utils import get_coco_api_from_dataset, CocoEvaluator
import train_utils.distributed_utils as utils

# mean_loss, lr = utils.train_one_epoch(model=model,optimizer=optimizer, data_loader=train_data_loader, device=device, epoch=epoch, print_freq=50)
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50, warmup=False):
    model.train()  # 切换到训练模式
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value: .6f}"))  # Q: 不明白
    header = "Epoch: [{}]".format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:
        warmup_factor = 5.0 / 10000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean loss
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = torch.stack(images, dim=0)

        boxes = []
        labels = []
        img_id = []
        for t in targets:
            boxes.append(t["boxes"])
            labels.append(t["labels"])
            img_id.append(t["image_id"])
        targets = {
            "boxes": torch.stack(boxes, dim=0),
            "labels": torch.stack(labels, dim=0),
            "image_id": torch.as_tensor(img_id)
        }

        # 注意: model, images, targets都会放入GPU
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        # 重要问题:
        # n个worker时, 是否每个worker取batch_size / n个图像组成batch?
        # n个GPU时, 是否同一个batch分布到n个GPU?或是n个batch分布到n个GPU, 
        # 从这里看, 似乎是同一个batch分布到n个GPU

        losses_dict = model(images, targets)
        losses = losses_dict["total_losses"]

        # reduce losses over all GPUs for logging purpose
        losses_dict_reduced = utils.reduce_dict(losses_dict)  # Q: 此处在多GPU训练时做debug
        losses_reduce = losses_dict_reduced["total_losses"]  # 多个tensor

        # loss_value = losses_reduce.debatch()  # Q: 没明白
        loss_value = losses_reduce.detach()
        mloss = (mloss * i + loss_value) / (i + 1)

        if not math.isfinite(loss_value):  # 当损失为无穷时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(losses_dict_reduced)
            sys.exit(1)
        
        # 假设存在前向计算图和后向计算图，前向计算图中存储用于反向传播时的中间变量
        # 后向计算图中存储梯度如dW，db等信息，那么
        # Q: optimizer.zero_grad()是清零后向后向计算图中的dW，db等信息吗？
        # A: debug查看: model.conf[0].weight, model.conf[0].weight.grad
        #    可以看到model.conf[0].weight.grad保存的dw数值不为0
        #    经过optimizer.zero_grad()后dw变为0
        optimizer.zero_grad()
        # Q: losses.backward()是计算后向计算图中的dW，db等信息吗？
        # A: 是调用losses对象中的grad_fn对象来计算dW, db
        #    debug查看model.conf[0].weight.grad, 可以看到有新的梯度保存
        # Q: losses只是一个数值，为什么还能有backward()方法？
        # A: 打印losses -> tensor(14.3723, grad_fn=<MeanBackward1>)
        #    可以看到losses不是一个普通的tensor对象, 该对象还会执行一个反向
        #    传播的函数, 来自ret = (total_loss * num_mask / pos_num).mean(dim=0)
        losses.backward()
        # Q: optimizer.step()是计算W - learning_rate * dW来更新W吗？
        # A: debug时可以通过如下步骤验证:
        #    A.1 w = model.conf[0].weight.clone()
        #    A.2 dw = model.conf[0].weight.grad.clone()
        #    A.3 new_w = w - optimizer.param_groups[0]["lr"] * dw
        #    A.4 在optimizer.step()之后对比model.conf[0].weight和new_w是否一致
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()  # 主要是更新learning_rate
        
        metric_logger.update(**losses_dict_reduced)  # Q: 在多GPU时查看
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
#################################################### 调试用, 正式代码删去 ################################################
        if i > 3:
            print("-----------> In train_eval_utils.py: 需删除!!!")
            break
#################################################### 调试用, 正式代码删去 ################################################
    return mloss, now_lr

# @torch.no_grad()是一个上下文管理器
# 等价于：
# model.eval()
# with torch.no_grad():
#     ...

# coco_info = utils.evaluate(model=model, data_loader=val_data_loader, device=device, data_set=val_data)
@torch.no_grad()
def evaluate(model, data_loader, device, data_set=None):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    # dataset提前做好，就不用每次都在这里执行一遍
    if data_set is None:
        dataset = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)  # Q: 是指xywh和ltrb吗？
    coco_evaluator = CocoEvaluator(data_set, iou_types)  # 根据iou_types来评估在data_set上的性能

#################################################### 调试用, 正式代码删去 ################################################
    debug_res_len = 0
#################################################### 调试用, 正式代码删去 ################################################

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = torch.stack(images, dim=0).to(device)  # Q: 为何放在cpu上评价

        # Q：初步理解这里torch.cuda.synchroize是等待所有的异步结束，更准确地计算推理的时间
        if device != torch.device("cpu"):
            torch.cuda.synchroize(device)
        
        model_time = time.time()
        results = model(images, targets=None)
        model_time = time.time() - model_time

        outputs = []
        for index, (bboxes_out, labels_out, scores_out) in enumerate(results):
            # height_width = targets["height_width"]
            height_width = targets[index]["height_width"]
            # 在get_coco_api_from_dataset中把box的坐标都转成了绝对坐标, 这里亦然
            bboxes_out[:, [0, 2]] = bboxes_out[:, [0, 2]] * height_width[1]
            bboxes_out[:, [1, 3]] = bboxes_out[:, [0, 2]] * height_width[0]

            info = {
                "boxes": bboxes_out.to(cpu_device),
                "labels": labels_out.to(cpu_device),
                "scores": scores_out.to(cpu_device)
            }
            outputs.append(info)

        # res:
        # {
        #     image_id_1:          {'boxes': tensor(...), 'labels': tensor(...), 'scores': tensor(...)},
        #     image_id_2:          {'boxes': tensor(...), 'labels': tensor(...), 'scores': tensor(...)},
        #     ...
        #     image_id_batch_size: {'boxes': tensor(...), 'labels': tensor(...), 'scores': tensor(...)}
        # }
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
#################################################### 调试用, 正式代码删去 ################################################
        debug_res_len += len(res)
        if debug_res_len > 8:
            break
#################################################### 调试用, 正式代码删去 ################################################

    # Q: synchronize_between_processes是同步进程数据，猜测是同步不同GPU上的数据
    metric_logger.synchronize_between_processes()
    print("Averaged stats: ", metric_logger)
    coco_evaluator.synchronize_between_processes()

    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()

    return coco_info

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    # Q: 上面部分代码完全没有作用
    iou_types = ["bbox"]
    return iou_types

