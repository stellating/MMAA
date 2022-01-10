import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from modules.dataloaders import R2DataLoader
import os

# IDX2CLS = {0: '恶性_乳头状癌',
#            1: '恶性_滤泡癌',
#            2: '恶性_髓样癌',
#            3: '恶性_其他',
#            4: '良性_结甲',
#            5: '良性_桥本甲状腺炎',
#            6: '良性_腺瘤',
#            7: '良性_其他',
#            8: '超声诊断_良性',
#            9: '超声诊断_良性可能性大',
#            10: '超声诊断_恶性',
#            11: '超声诊断_恶性可能性大',
#            12: '超声诊断_不能确定',
#            13: '结构_实性',
#            14: '结构_囊实性且实性部分偏心',
#            15: '结构_囊实性且实性部分不偏心',
#            16: '结构_海绵征',
#            17: '结构_囊性',
#            18: '大小_＜0.5cm',
#            19: '大小_0.5-1cm',
#            20: '大小_1-1.5cm',
#            21: '大小_1.5-2cm',
#            22: '大小_2-3cm',
#            23: '大小_＞3cm',
#            24: '纵横比_＜1',
#            25: '纵横比_≥1',
#            26: '形态_规则',
#            27: '形态_不规则',
#            28: '边界_清晰',
#            29: '边界_模糊',
#            30: '边缘_无特殊',
#            31: '边缘_边缘小分叶',
#            32: '边缘_毛刺',
#            33: '边缘_边缘小分叶及毛刺',
#            34: '晕_无',
#            35: '晕_规则细晕',
#            36: '晕_不规则晕',
#            37: '回声水平_高',
#            38: '回声水平_中等',
#            39: '回声水平_低',
#            40: '回声水平_极低',
#            41: '回声均一性_均匀',
#            42: '回声均一性_不均匀',
#            43: '钙化形态_微钙化',
#            44: '钙化形态_其他钙化',
#            45: '钙化形态_微钙化及边缘钙化中断伴低回声突出钙化外',
#            46: '钙化形态_微钙化及其他钙化',
#            47: '钙化形态_无钙化',
#            48: '部位_1_周边',
#            49: '部位_2_内部',
#            50: '杂乱_是',
#            51: '杂乱_否',
#            52: '穿支_有',
#            53: '穿支_无',
#            54: '局限性丰富_有',
#            55: '局限性丰富_无',
#            56: '血管走行_规则',
#            57: '血管走行_不规则',
#            58: '侵犯被膜_是',
#            59: '部位_A_左叶',
#            60: '部位_A_右叶',
#            61: '部位_A_峡部左侧',
#            62: '部位_A_峡部正中',
#            63: '部位_A_峡部右侧',
#            64: '部位_B_上部',
#            65: '部位_B_上中部',
#            66: '部位_B_上、中部',
#            67: '部位_B_中部',
#            68: '部位_B_中下部',
#            69: '部位_B_中、下部',
#            70: '部位_B_下部',
#            71: '部位_B_上、中、下部',
#            72: '部位_C_血管侧',
#            73: '部位_C_正中',
#            74: '部位_C_气管侧',
#            75: '与被膜关系_远离被膜',
#            76: '与被膜关系_紧邻被膜',
#            77: '与被膜关系_被膜连续性中断',
#            78: '与被膜关系_突出被膜外',
#            79: '与气管关系_远离',
#            80: '与气管关系_紧邻',
#            81: '与气管关系_分界不清',
#            82: '特殊征象_彗星征',
#            83: '血流丰富_是',
#            84: '血流丰富_否'}
IDX2CLS={}
import json
with open('/media/gyt/00eebf84-091a-4eed-82b7-3f2c69ba217b/1-data/IUxray_data/iu_xray/iu_category_40.json','r') as j:
    lines = json.load(j)
for c,line in lines.items():
    IDX2CLS[line]=c



class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            print(name)
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x['0']]
        return outputs, x['0']

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.label_extractor.backbone._modules['0'], target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        print(target_activations)
        print(output.shape)
        output = output.view(output.size(0), -1)
        print(output.shape)
        output = self.model.classifier(output)
        return target_activations, output

def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input) 

    def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_variables=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_variables=True)

        output = input.grad.cpu().data.numpy()
        output = output[0,:,:,:]

        return output

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def print_top_classes(predictions, **kwargs):
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(10, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(IDX2CLS[cls_idx])
        if len(IDX2CLS[cls_idx]) > max_str_len:
            max_str_len = len(IDX2CLS[cls_idx])

    print('Top 10 classes:')
    output_str=''
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, IDX2CLS[cls_idx])
        output_string += ' ' * (max_str_len - len(IDX2CLS[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)
        output_str+=output_string
    return output_str

# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    print('R_ss:',R_ss.shape,cam_ss.shape)
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    print('R_ss:',R_ss.shape)
    return R_ss_addition

def generate_relevance(model, input, index=None):
    output,_ = model(input, mode='sample')
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)


    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot_vector = one_hot
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

#     num_tokens = model.blocks[0].attn.get_attention_map().shape[-1]
    print(model.label_extractor.transformer.encoder)
    num_tokens = model.label_extractor.transformer.encoder.layers[0].get_attention_map().shape
    print('num_tokens:',num_tokens)
    num_tokens = num_tokens[1]
    CAM = torch.zeros(144).cuda()
    # R = torch.eye(num_tokens, num_tokens).cuda()
    for blk in model.label_extractor.transformer.encoder.layers:
        grad = blk.get_attn_gradients()
        # print(grad.shape)
        cam = blk.get_attention_map()
        cam = avg_heads(cam, grad)

        CAM += cam[index,:][0]
        print('CAM:', CAM.shape)

        # R1 = apply_self_attention_rules(R.cuda(), cam.cuda())
        # print('r.shape:', R.shape, R1.shape)
        # R += R1
    return CAM

def generate_visualization(model,original_image, class_index=None):
    transformer_attribution = generate_relevance(model, original_image.cuda(), index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 12, 12)
    print(transformer_attribution.shape)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=32, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(384, 384).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    print(original_image.shape)
    image = torch.squeeze(original_image, dim=0)
    print('imageshape:',image.shape)
    image_transformer_attribution = image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    print(image_transformer_attribution.shape,transformer_attribution.shape)
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    from models.r2genq2l import R2GenModel
    from modules.tokenizers import Tokenizer
    import json
    import argparse

    config_file = 'config.json'
    with open(config_file, 'r') as j:
        dic = json.load(j)
    args = argparse.Namespace(**dic)

    args.batch_size = 1


    print(args)

    tokenizer = Tokenizer(args)

    model = R2GenModel(args, tokenizer).cuda()

    resume_path = str(args.resume)
    print("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    output_file = open('output.txt','w')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    # image = Image.open(
    #     '/media/gyt/00eebf84-091a-4eed-82b7-3f2c69ba217b/my-caption/Thyroid-ML-GCN/data/ThyroidImage2021/008428.jpg').convert(
    #     'RGB')
    # image = Image.open('/media/gyt/00eebf84-091a-4eed-82b7-3f2c69ba217b/1-data/IUxray_data/iu_xray/images/CXR2_IM-0652/0.png').convert('RGB')
    # print(image)

    for batch_idx, (images_id, images, labels, reports_ids, reports_masks) in enumerate(test_dataloader):
        images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
        fc_feats, output = model(images, mode='sample')
        reports = model.tokenizer.decode_batch(output.cpu().numpy())
        ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
        print('reports:',reports)
        print('ground truth:',ground_truths)
        class_string = print_top_classes(fc_feats)
        gt_class_string = print_top_classes(labels)
        print(images_id[0])
        output_file.write(images_id[0]+'|'+reports[0]+'|'+ground_truths[0]+'|'+class_string+'|'+gt_class_string+'\n')
        # img = Image.open(os.path.join(args.image_dir, images_id[0])).convert('RGB')
        # fig, axs = plt.subplots(1, 2)
        #
        #
        # axs[0].imshow(img);
        # axs[0].axis('off');
        #
        # cat = generate_visualization(model, images)
        # axs[1].imshow(cat);
        # axs[1].axis('off');
        # main_path = os.path.dirname('cam_results_test/'+images_id[0]+'_cam.png')
        # if not os.path.exists(main_path):#如果路径不存在
        #     os.makedirs(main_path)
        # plt.savefig('cam_results_test/'+images_id[0]+'_cam.png', bbox_inches='tight')


    output_file.close()
    # Can work with any model, but it assumes that the model has a 
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    # grad_cam = GradCam(model = model, \
    #                 target_layer_names = ["layer4"], use_cuda=args.use_cuda)
    #########################################################################################

    #
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(image);
    # axs[0].axis('off');
    #
    #
    # image = torch.unsqueeze(thyroid_image, dim=0)
    # fc_feats, output = model(image, mode='sample')
    # reports = model.tokenizer.decode_batch(output.cpu().numpy())
    # print(reports)
    # print_top_classes(fc_feats)
    # # cat - the predicted class
    # print('input_proj:',model.label_extractor.input_proj)
    # cat = generate_visualization(model,thyroid_image)
    #
    # axs[1].imshow(cat);
    # axs[1].axis('off');
    # plt.show()
    #############################################################################################

    # transformer = model.label_extractor.transformer
    # x = model(image,mode='sample')
    # print(x)

    # axs[1].imshow(cat);
    # axs[1].axis('off');
    # axs[2].imshow(dog);
    # axs[2].axis('off');

    # # If None, returns the map for the highest scoring category.
    # # Otherwise, targets the requested index.
    # target_index = 13
    #
    # mask = grad_cam(input=image,index=target_index)
    #
    # show_cam_on_image(image, mask)
    #
    # gb_model = GuidedBackpropReLUModel(model = models.vgg19(pretrained=True), use_cuda=args.use_cuda)
    # gb = gb_model(input=image, index=target_index)
    # utils.save_image(torch.from_numpy(gb), 'gb.jpg')
    #
    # cam_mask = np.zeros(gb.shape)
    # for i in range(0, gb.shape[0]):
    #     cam_mask[i, :, :] = mask
    #
    # cam_gb = np.multiply(cam_mask, gb)
    # utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')

