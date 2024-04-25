import os.path as osp
import os
import random
from copy import deepcopy
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import json
from randaugment import RandAugment
from PIL import Image
import aug_lib
from data_utils.pa100k_dataset import PedesAttr
from utils.cutout import SLCutoutPIL

# YOUR_PATH/MyProject/others_prj/query2labels/data/intentonomy
inte_image_path = '/data/sqhy_data/intent_resize'
inte_train_anno_path = '/data/sqhy_data/intent_resize/annotations/intentonomy_train2020.json'
inte_val_anno_path = '/data/sqhy_data/intent_resize/annotations/intentonomy_val2020.json'
inte_test_anno_path = '/data/sqhy_data/intent_resize/annotations/intentonomy_test2020.json'

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

def process_labels(target):
    """
    Encode multiple labels using 1/0 encoding

    Args:
        target: xml tree file
    Returns:
        torch tensor encoding labels as 1/0 vector
    """

    ls = target['annotation']['object']

    j = []
    if type(ls) == dict:
        if int(ls['difficult']) == 0:
            j.append(object_categories.index(ls['name']))

    else:
        for i in range(len(ls)):
            if int(ls[i]['difficult']) == 0:
                j.append(object_categories.index(ls[i]['name']))

    k = np.zeros(len(object_categories))
    k[j] = 1

    return torch.from_numpy(k)



class InteDataSet(data.Dataset):
    def __init__(self, 
                 image_dir, 
                 anno_path, 
                 input_transform=None, 
                 labels_path=None,
    ):
        self.image_dir = image_dir
        self.anno_path = anno_path
        
        self.input_transform = input_transform
        self.labels_path = labels_path
        
        self.labels = []
        if os.path.exists(self.labels_path):
            self.labels = np.load(self.labels_path).astype(np.float64)
            self.labels = (self.labels > 0).astype(np.float64)
        else:
            print('labels_path not exist, please check the path or run get_label_vector.py first')
    
    def _load_image(self, index):
        image_path = self._get_image_path(index)
        
        return Image.open(image_path).convert("RGB")
    
    def _get_image_path(self, index):
        with open(self.anno_path, 'r') as f:
            annos_dict = json.load(f)
            annos_i = annos_dict['annotations'][index]
            id = annos_i['id']
            if id != index:
                raise ValueError('id not equal to index')
            img_id_i = annos_i['image_id']
            
            imgs = annos_dict['images']
            
            for img in imgs:
                if img['id'] == img_id_i:
                    image_file_name = img['filename']
                    image_file_path = os.path.join(self.image_dir, image_file_name)
                    break
        
        return image_file_path
                    
    def __getitem__(self, index):
        input = self._load_image(index)
        if self.input_transform:
            input = self.input_transform(input)
            # print(self.input_transform.transform1, self.input_transform.transform2)

        label = self.labels[index]
        return input, label
    
    def __len__(self):
        return self.labels.shape[0]

class TwoCropTransform:
    # https://github.com/HobbitLong/SupContrast/blob/331aab5921c1def3395918c05b214320bef54815/util.py#L9
    """Create two augmentations of the same image"""
    def __init__(self, transform1,transform2):

        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):

        return [self.transform1(x), self.transform2(x)]

        # return [trans1(x), trans2(x)]

def get_datasets(args):
    trivialaugment = aug_lib.TrivialAugment()
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    transform_list1 = [
        transforms.Resize((args.img_size_hight, args.img_size_weight)),
        SLCutoutPIL(n_holes=args.n_holes, length=args.length),
        RandAugment(),
        transforms.ToTensor(),
        normalize
    ]
    transform_list = transforms.Compose(transform_list1)

    test_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size_hight, args.img_size_weight)),
                                            transforms.ToTensor(),
                                            normalize])
    
    if args.dataname == 'intentonomy':
        # ! config your data path here.
        dataset_dir = args.dataset_dir

        train_dataset = InteDataSet(
            image_dir=inte_image_path,
            anno_path=inte_train_anno_path,
            input_transform=TwoCropTransform(transform_list, transform_list),
            labels_path='/home/shiqinghongya/uncertainty/data/intentonomy/train_label_vectors_intentonomy2020.npy',
        )
        val_dataset = InteDataSet(
            image_dir=inte_image_path,
            anno_path=inte_val_anno_path,
            input_transform=test_data_transform,
            labels_path='/home/shiqinghongya/uncertainty/data/intentonomy/val_label_vectors_intentonomy2020.npy',
        )
        test_dataset = InteDataSet(
            image_dir=inte_image_path,
            anno_path=inte_test_anno_path,
            input_transform=test_data_transform,
            labels_path='/home/shiqinghongya/uncertainty/data/intentonomy/test_label_vectors_intentonomy2020.npy',
        )
    elif args.dataname == 'PA100K':
        train_dataset=PedesAttr(cfg=args, split='trainval', transform=TwoCropTransform(transform_list, transform_list))
        val_dataset = PedesAttr(cfg=args, split='test', transform=test_data_transform)
        test_dataset = PedesAttr(cfg=args, split='test', transform=test_data_transform)
    elif args.dataname == 'RAP':

        train_dataset = PedesAttr(cfg=args, split='trainval', transform=TwoCropTransform(transform_list, transform_list))
        val_dataset = PedesAttr(cfg=args, split='test', transform=test_data_transform)
        test_dataset = PedesAttr(cfg=args, split='test', transform=test_data_transform)
    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)


    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    print("len(test_dataset):", len(test_dataset))

    return train_dataset, val_dataset, test_dataset







