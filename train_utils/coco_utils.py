
import torch
import torchvision
import torch.utils.data
from pycocotools.coco import COCO

"""
my_dataset.VOCDataSet(ds) vs. pycocotools.coco.COCO(coco_ds)
------------------------------------------------------------
ds.class_dict
ds.img_root
ds.root
ds.transforms
ds.xml_list
ds.coco_index(img_idx) 注: 根据img_idx返回对应的target

coco_ds
coco_ds.dataset
coco_ds.anns
coco_ds.catToImgs
coco_ds.cats
coco_ds.imgToAnns
coco_ds.imgs
"""
def convert_to_coco_api(ds):
    coco_ds = COCO()
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        targets = ds.coco_index(img_idx)  # my_dataet中的coco()函数
        image_id = targets["image_id"].item()  # Q: item()是一个迭代器吗？
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = targets["height_width"][0]
        img_dict["width"] = targets["height_width"][1]
        dataset["images"].append(img_dict)

        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * img_dict["width"]
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * img_dict["height"]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = (targets["area"] * img_dict["width"] * img_dict["height"]).tolist()
        iscrowd = targets["iscrowd"].tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    # 基于coco_ds.dataset生成:
    # coco_ds.anns, coco_ds.imgToAnns, coco_ds.catToImgs, coco_ds.imgs, coco_ds.cats
    coco_ds.createIndex()
    return coco_ds

def get_coco_api_from_dataset(dataset):
    for _ in range(10):  # Q: 为何迭代10次？
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):  # Q: 没看明白
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)
