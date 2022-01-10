import os
import json
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
import numpy as np
import os
import random

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.args = args
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)
    

class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        self.num_classes = self.args.num_class
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        labels = example['new_label']
        labels_index = sorted(labels)
        labels = np.zeros(self.num_classes, np.float32)
        labels[labels_index] = 1
        labels = torch.Tensor(labels)

        sample = (image_path[0], image, labels, report_ids, report_masks, seq_length)
        return sample

class IuxraySingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        self.num_classes = self.args.num_class
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image_1)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        labels = example['new_label']
        labels_index = sorted(labels)
        labels = np.zeros(self.num_classes, np.float32)
        labels[labels_index] = 1
        labels = torch.Tensor(labels)

        sample = (image_path, image, labels, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        self.num_classes = 881
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        file_path = os.path.join(self.image_dir, image_path[0])
#         if os.path.isfile(file_path): 
        image = Image.open(file_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        labels = example['label']
        labels_index = sorted(labels)
        labels = np.zeros(self.num_classes, np.float32)
        labels[labels_index] = 1
        labels = torch.Tensor(labels)
        sample = (image_id, image, labels, report_ids, report_masks, seq_length)
        return sample

class ThyroidDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        coco_root = args.ann_path
        self.root = coco_root
        self.img_root = os.path.join(coco_root, 'ThyroidImage2021')
        self.phase = split
        self.img_list = []
        self.get_anno()
        self.num_classes = 85
        self.transform = transform
        self.max_seq_length = args.max_seq_length
        self.tokenizer = tokenizer
        self.epoch = 1

    def get_anno(self):
        list_path = os.path.join(self.root, '{}_anno_caption.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root, 'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        item = self.img_list[index]

        return self.get(item)

    def get(self, item):
        image_id = item['filename']
        # print(image_ID)
        img_name = os.path.join(self.img_root, image_id)
        image = Image.open(img_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        labels_index = sorted(item['labels'])
        labels = np.zeros(self.num_classes, np.float32)
        labels[labels_index] = 1
        labels = torch.Tensor(labels)


        report = item['report'][:self.max_seq_length]
        report_ids = self.tokenizer(report)
        report_masks = [1]*len(report_ids)
        seq_length = len(report_ids)
        sample = (image_id, image, labels, report_ids, report_masks, seq_length)
        return sample
