import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from torch.autograd import Variable
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

import numpy as np
import tqdm
import json
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from base_dirs import *

def parse_args():
    parser = argparse.ArgumentParser(description='Test the data and save the raw detections')
    parser.add_argument('--dataset', default = 'voc', help='voc or coco')
    parser.add_argument('--subset', default = None, help='train or val or test')
    parser.add_argument('--dir', default = None, help='directory of object detector weights')
    parser.add_argument('--checkpoint', default = 'latest.pth', help='what is the name of the object detector weights')
    parser.add_argument('--saveNm', default = None, help='name to save results as')
    args = parser.parse_args()
    return args

args = parse_args()


#load the config file for the model that will also return logits
if args.dataset == 'voc':
    args.config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712OS_wLogits.py'
else:
    args.config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_cocoOS_wLogits.py'
    
###################################################################################################
##############Setup Config file ###################################################################
cfg = Config.fromfile(args.config)

# import modules from string list.
if cfg.get('custom_imports', None):
    from mmcv.utils import import_modules_from_strings
    import_modules_from_strings(**cfg['custom_imports'])
# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True
cfg.model.pretrained = None
if cfg.model.get('neck'):
    if isinstance(cfg.model.neck, list):
        for neck_cfg in cfg.model.neck:
            if neck_cfg.get('rfp_backbone'):
                if neck_cfg.rfp_backbone.get('pretrained'):
                    neck_cfg.rfp_backbone.pretrained = None
    elif cfg.model.neck.get('rfp_backbone'):
        if cfg.model.neck.rfp_backbone.get('pretrained'):
            cfg.model.neck.rfp_backbone.pretrained = None

# in case the test dataset is concatenated
if isinstance(cfg.data.testOS, dict):
    cfg.data.testOS.test_mode = True
elif isinstance(cfg.data.testOS, list):
    for ds_cfg in cfg.data.testOS:
        ds_cfg.test_mode = True

distributed = False

samples_per_gpu = cfg.data.testOS.pop('samples_per_gpu', 1)
if samples_per_gpu > 1:
    # Replace 'ImageToTensor' to 'DefaultFormatBundle'
    cfg.data.testOS.pipeline = replace_ImageToTensor(cfg.data.testOS.pipeline)

###################################################################################################
###############Load Dataset########################################################################
print("Building datasets")
if args.dataset == 'voc':
    num_classes = 15
    if args.subset == 'train12':
        dataset = build_dataset(cfg.data.trainCS12)
    elif args.subset == 'train07':
        dataset = build_dataset(cfg.data.trainCS07)
    elif args.subset == 'val':
        dataset = build_dataset(cfg.data.val)
    elif args.subset == 'test':
        dataset = build_dataset(cfg.data.testOS)
    else:
        print('That subset is not implemented.')
        exit()
else:
    if args.subset == 'train':
        dataset = build_dataset(cfg.data.trainCS)
    elif args.subset == 'val':
        dataset = build_dataset(cfg.data.val)
    elif args.subset == 'test':
        dataset = build_dataset(cfg.data.testOS)
    else:
        print('That subset is not implemented.')
        exit()

    if args.dataset == 'coco':
        num_classes = 50
    else:
        #for the full version of coco used to fit GMMs in the iCUB experiments
        num_classes = 80


data_loader = build_dataloader(
    dataset,
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=distributed,
    shuffle=False)


###################################################################################################
###############Build model ########################################################################
print("Building model")

# build the model and load checkpoint
model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
method_list = [func for func in dir(model) if callable(getattr(model, func))]
fp16_cfg = cfg.get('fp16', None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
checkpoint = load_checkpoint(model, '{}/{}/{}'.format(BASE_WEIGHTS_FOLDER, args.dir, args.checkpoint), map_location='cpu')

if 'CLASSES' in checkpoint['meta']:
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    model.CLASSES = dataset.CLASSES

model = MMDataParallel(model, device_ids=[0])
model.eval()


########################################################################################################
########################## TESTING DATA  ###############################################################
########################################################################################################
print(f"Testing {args.subset} data")
num_images = len(data_loader.dataset)

score_threshold = 0.2 # only detections with a max softmax above this score are considered valid
total = 0
allResults = {}
for i, data in enumerate(tqdm.tqdm(data_loader, total = num_images)):   
    imName = data_loader.dataset.data_infos[i]['filename']
    # print(f'Image Name: {imName}')

    allResults[imName] = []

    total += 1
    all_detections = None
    all_scores = []
    
    # with torch.no_grad():
    result = model(return_loss = False, rescale=True, **data)[0]  

    #collect results from each class and concatenate into a list of all the results (15 classes)
    for j in range(50):
        dets = result[j]

        if len(dets) == 0:
            continue

        bboxes = dets[:, :5].detach().cpu().numpy()
        dists = dets[:, 5:]    # unnormalised distribution values for all classes associated with detection
        scores = dets[:, 4].detach().cpu().numpy()        # confidence scores for all classes associated with detection
        confs = []      # Gradnorm variable
        scoresT = np.expand_dims(scores, axis=1)

        dists_detached = dists.detach().cpu().numpy()
        #winning class must be class j for this detection to be considered valid
        mask = np.argmax(dists_detached, axis = 1)==j    #Check validity of detection

        if np.sum(mask) == 0:
            continue

        #check thresholds are above the score cutoff (ensure detections above confidence score)
        imDets = np.concatenate((dists_detached, bboxes, scoresT), 1)[mask]
        scores = scores[mask]
        mask2 = scores[::-1] >= score_threshold

        # Conver mask2 from true, false -> indices of true values
        mask2_indexs = np.where(mask2)[0]

        # If no detections are above threshold, continue loop
        if np.sum(mask2) == 0:
            continue
        
        imDets = imDets[mask2]

        # apply GradNorm
        for detection in dists[mask2_indexs,:]:
            
            # Softmax initialisation 
            logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
            targets = torch.ones((1, len(detection))).cuda() # targets for gradnorm algo
            loss = torch.mean(torch.sum(-targets * logsoftmax(detection), dim=-1))

            loss.backward(retain_graph=True)

            layer_grad = model.module.roi_head.bbox_head.fc_cls.weight.grad
            layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
            confs.append(layer_grad_norm)
            model.zero_grad()

        # print(f'Number of boxes: {len(bboxes)}')
        # print(f'boxes: {bboxes}')
        # print(f'Distributions: {dists}')
        # print(f'Scores: {scores}')
        # print(f'ScoresT: {scoresT}')
        # print(f'confidences: {confs}')
        imDets = np.concatenate((imDets, np.expand_dims(confs, axis=1)), 1)

        # Initialise the variable 
        if all_detections is None:
            all_detections = imDets

        # Add to exisiting variable after first detection results
        else:
            all_detections = np.concatenate((all_detections, imDets))

    if all_detections is None:
        continue
    else:

        #remove doubled-up detections -- this shouldn't really happen
        detections, idxes = np.unique(all_detections, return_index = True, axis = 0)

    allResults[imName] = detections.tolist()

    # print(allResults[imName])
    # break

#save results
jsonRes = json.dumps(allResults)

save_dir = f'{BASE_RESULTS_FOLDER}/FRCNN/raw/{args.dataset}/{args.subset}'
#check folders exist, if not, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

f = open('{}/{}.json'.format(save_dir, args.saveNm), 'w')
f.write(jsonRes)
f.close()

