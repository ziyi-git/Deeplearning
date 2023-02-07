import os

from torch.utils.data import Dataset
import torch
import json
from lxml import etree
from PIL import Image

class VOCDataSet(Dataset):
    """
    解析PASCAL VOC2007/2012数据集
    """
    def __init__(self, voc_root, year="2012", transforms=None, train_set="train.txt"):
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']."

        if 'VOCdevkit' in voc_root:
            self.root = os.path.join(voc_root, f"VOC{year}")
        else:
            self.root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")

        txt_list = os.path.join(self.root, "ImageSets", "Main", train_set)

        with open(txt_list) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines() if len(line.strip()) > 0]
        
        json_file = "./pascal_voc_classes.json"
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file, "r") as f:
            self.class_dict = json.load(f)
        
        self.transforms = transforms
    
    def __len__(self):
        return len(self.xml_list)
    
    # __getitem__讲解: https://zhuanlan.zhihu.com/p/27661382
    def __getitem__(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:  # fid: _io.TextIOWrapper
            xml_str = fid.read()  # '<annotation>\n\t<folder>VOC2012...'
        xml = etree.fromstring(xml_str)  # tag: "annotation"
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        height_width = [data_height, data_width]
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG.".format(img_path))
        
        assert "object" in data, "{} lack of object information.".format(xml_path)
        boxes = []
        labels = []
        iscrowed = []  # 是否重叠
        for obj in data["object"]:
            # 对原始标注的box归一化
            xmin = float(obj["bndbox"]["xmin"]) / data_width
            xmax = float(obj["bndbox"]["xmax"]) / data_width
            ymin = float(obj["bndbox"]["ymin"]) / data_height
            ymax = float(obj["bndbox"]["ymax"]) / data_height
        
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <= 0.".format(xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowed.append(int(obj["difficult"]))
            else:
                iscrowed.append(0)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)  # torch.Size([nboxes, 4])
        labels = torch.as_tensor(labels, dtype=torch.int64)  # torch.Size([nboxes])
        iscrowed = torch.as_tensor(iscrowed, dtype=torch.int64)
        height_width = torch.as_tensor(height_width, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # torch.Size([nboxes])

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id  # 查询时的idx
        target['area'] = area
        target['iscrowed'] = iscrowed
        target['height_width'] = height_width

        if self.transforms is not None:
            # 在AssignGTtoDefaultBox函数:
            # 1. 将target['boxes']由torch.Size([nboxes, 4])变为torch.Size([8732, 4])
            # 2. 将target['boxes']的格式由[xmin, ymin, xmax, ymax]变为[x, y, w, h](encode函数)
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml_str)['annotation']
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width
    
    def parse_xml_to_dict(self, xml):
        # 最底层的返回条件: 如return {'xmin': '53'}
        if len(xml) == 0:
            return {xml.tag: xml.text}
        
        # 次底层的遍历: 如return {'bndbox': {'xmin': '53', 'ymin': '87', 'xmax': '471', 'ymax': '420'}}
        # 最底层->次底层->...->顶层({'annotation': ....})
        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}
    
    def coco_index(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        height_width = [data_height, data_width]
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"]) / data_width
            xmax = float(obj["bndbox"]["xmax"]) / data_width
            ymin = float(obj["bndbox"]["ymin"]) / data_height
            ymax = float(obj["bndbox"]["ymax"]) / data_height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        height_width = torch.as_tensor(height_width, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["height_width"] = height_width

        return target
    
    # 静态方法可以在不创建类实例的情况下调用(执行效率高?)
    # 例如:
    # train_dataset = VOCDataSet(VOC_root, '2012', data_transform['train'], train_set='train.txt')
    # 1. train_dataset.collate_fn(batch)
    # 2. VOCDataSet.collate_fn(batch)
    # 上述2种方法的效果一样
    @staticmethod
    def collate_fn(batch):
        # 假设batch的形式如下:
        # batch = [(img1, target1), (img2, target2), (img3, target3), (img4, target4)]
        # *batch相当于对列表解包, 形式如下:
        # (img1, target1)
        # (img2, target2)
        # (img3, target3)
        # (img4, target4)
        # zip(*batch)分别对img和target进行压缩
        # 同时使用tuple进行解包, 形式如下:
        # ((img1, img2, img3, img4), (target1, target2, target3, target4))
        # 所以:
        # images = (img1, img2, img3, img4)
        # targets = (target1, target2, target3, target4)
        images, targets = tuple(zip(*batch))
        
        return images, targets
