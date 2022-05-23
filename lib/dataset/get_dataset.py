import torchvision.transforms as transforms
from dataset.cocodataset import CoCoDataset
from dataset.thyroiddataset import ThyroidDataset
from dataset.iuxraydataset import IUXrayDataset
# from dataset.heartTUdataset import heartTUDataset
import os

def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    trainTransform = transforms.Compose( [transforms.Resize((args.img_size, args.img_size)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normTransform])

    testTransform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                        transforms.CenterCrop(args.img_size),
                                        transforms.ToTensor(),
                                        normTransform])

    test_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])
    

    if args.dataname == 'coco' or args.dataname == 'coco14':
        # ! config your data path here.
        val_dataset = CoCoDataset(
            image_dir='/media/gyt/00eebf84-091a-4eed-82b7-3f2c69ba217b/code/data/coco/data/val2014',
            anno_path='/media/gyt/00eebf84-091a-4eed-82b7-3f2c69ba217b/code/data/coco/annotations_trainval2014/annotations/instances_val2014.json',
            input_transform=test_data_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
        )
    elif args.dataname == 'thyroid':
        coco_root = 'data/thyroid/thyroid_annotation'
        image_root = 'data/thyroid/'
        train_img_root = os.path.join(image_root, 'ThyroidImage2021')
        test_img_root = os.path.join(image_root, 'ThyroidImage2021')
        train_dataset = ThyroidDataset(
            split='train',
            num_labels=args.num_class,
            root=coco_root,
            img_root=train_img_root,
            transform=trainTransform,
            testing=False)
        val_dataset = ThyroidDataset(split='val',
                                num_labels=args.num_class,
                                root=coco_root,
                                img_root=test_img_root,
                                transform=testTransform,
                                testing=True)
        test_dataset = ThyroidDataset(split='test',
                               num_labels=args.num_class,
                               root=coco_root,
                               img_root=test_img_root,
                               transform=testTransform,
                               testing=True)
    elif args.dataname == 'iuxray':
        coco_root = '/user-data/mydata/IU-Xray/'
        img_root = '/user-data/mydata/IU-Xray/'
        train_img_root = os.path.join(img_root, 'images/images_normalized')
        test_img_root = os.path.join(img_root, 'images/images_normalized')
        train_dataset = IUXrayDataset(
            split='train',
            num_labels=args.num_class,
            root=coco_root,
            img_root=train_img_root,
            transform=trainTransform,
            testing=False)
        val_dataset = IUXrayDataset(split='val',
                                num_labels=args.num_class,
                                root=coco_root,
                                img_root=test_img_root,
                                transform=testTransform,
                                testing=True)
        test_dataset = IUXrayDataset(split='test',
                               num_labels=args.num_class,
                               root=coco_root,
                               img_root=test_img_root,
                               transform=testTransform,
                               testing=True)
#     elif args.dataname == 'heart':
#         coco_root = '/userhome/gyt/data/heartTU/'
#         img_root = '/userhome/gyt/data/heartTU/heartTU_images'
#         train_img_root = img_root
#         test_img_root = img_root
#         train_dataset = heartTUDataset(
#             split='train',
#             num_labels=args.num_labels,
#             root=coco_root,
#             img_root=train_img_root,
#             transform=trainTransform,
#             testing=False)
#         val_dataset = heartTUDataset(split='val',
#                                 num_labels=args.num_labels,
#                                 root=coco_root,
#                                 img_root=test_img_root,
#                                 transform=testTransform,
#                                 testing=True)
#         test_dataset = heartTUDataset(split='test',
#                                num_labels=args.num_labels,
#                                root=coco_root,
#                                img_root=test_img_root,
#                                transform=testTransform,
#                                testing=True)
    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)
        
    print("len(val_dataset):", len(val_dataset))
    return train_dataset,val_dataset
