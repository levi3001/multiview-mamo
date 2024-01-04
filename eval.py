"""
Run evaluation on a trained model to get mAP and class wise AP.

USAGE:
python eval.py --config data_configs/voc.yaml --weights outputs/training/fasterrcnn_convnext_small_voc_15e_noaug/best_model.pth --model fasterrcnn_convnext_small
"""
from datasets import (
    create_valid_dataset, create_valid_loader
)

from datasets_DDSM import(
    create_train_dataset_DDSM, create_valid_dataset_DDSM, create_test_dataset_DDSM
)
from models.create_fasterrcnn_model import create_model

from torch_utils import utils
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
from tqdm import tqdm
import os

import torch
import argparse
import yaml
import torchvision
import time
import numpy as np
from torch_utils import froc
torch.multiprocessing.set_sharing_strategy('file_system')


@torch.inference_mode()
def evaluate(
    model, 
    data_loader, 
    device, 
    num_classes =2,
    out_dir=None,
    classes=None,
    colors=None
):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    target = []
    preds = []
    counter = 0
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        counter += 1
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        with torch.no_grad():
            outputs = model(images)
            #print('out',outputs)
            #print('tar',targets)
        #####################################
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
        #####################################

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    torch.set_num_threads(n_threads)
    #metric = MeanAveragePrecision(class_metrics=args['verbose'], iou_thresholds =[0.2])
    metric = froc.FROC(num_classes, classes, threshold=[0.25,0.5,1,2,4], view ='all')
    #metric.update(preds, target)
    #metric_summary = metric.compute()
    metric_summary = metric.compute(preds,target)
    return metric_summary
if __name__ == '__main__':
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', 
        default='data_configs/test_image_config.yaml',
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', 
        default='fasterrcnn_resnet50_fpn',
        help='name of the model'
    )
    parser.add_argument(
        '-mw', '--weights', 
        default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-ims', '--imgsz', 
        nargs='+',
        default=(1400, 1700), 
        type=int, 
        help='image size to feed to the network'
    )
    parser.add_argument(
        '-w', '--workers', default=4, type=int,
        help='number of workers for data processing/transforms/augmentations'
    )
    parser.add_argument(
        '-b', '--batch', 
        default=8, 
        type=int, 
        help='batch size to load the data'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )

    parser.add_argument(
        '--norm', 
        default=None,
        help='normalization type'
    )
    parser.add_argument(
        '--data-dir', 
        default='../',
        help='data directory'
    )
    args = vars(parser.parse_args())

    # Load the data configurations
    with open(args['data']) as file:
        data_configs = yaml.safe_load(file)

    # Validation settings and constants.
    try: # Use test images if present.
        VALID_DIR_IMAGES = os.path.normpath(data_configs['TEST_DIR_IMAGES'])
        VALID_DIR_LABELS = os.path.normpath(data_configs['TEST_DIR_LABELS'])
    except: # Else use the validation images.
        VALID_DIR_IMAGES = os.path.normpath(data_configs['VALID_DIR_IMAGES'])
        VALID_DIR_LABELS = os.path.normpath(data_configs['VALID_DIR_LABELS'])
    DATA_DIR= args['data_dir']
    VALID_DIR_IMAGES = os.path.join(DATA_DIR, VALID_DIR_IMAGES)    
    VALID_DIR_LABELS = os.path.join(DATA_DIR, VALID_DIR_LABELS)
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']
    NUM_WORKERS = args['workers']
    DEVICE = args['device']
    BATCH_SIZE = args['batch']
    dataset_name = data_configs['DATASET']
    # Model configurations
    IMAGE_SIZE = args['imgsz']

    # Load the pretrained model
    create_model = create_model[args['model']]
    if args['weights'] is None:
        try:
            model, coco_model = create_model(num_classes=NUM_CLASSES, norm = args['norm'],  size= IMAGE_SIZE, coco_model=True)
        except:
            model = create_model(num_classes=NUM_CLASSES, norm = args['norm'], size= IMAGE_SIZE, coco_model=True)

    # Load weights.
    if args['weights'] is not None:
        model = create_model(num_classes=NUM_CLASSES, norm= args['norm'], size= IMAGE_SIZE, coco_model=False)
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        checkpoint1 = {}
        for key in checkpoint['model_state_dict']:
            if key.split('.')[0] == 'module':
                checkpoint1[key[7:]]= checkpoint['model_state_dict'][key]
            else:
                checkpoint1[key] = checkpoint['model_state_dict'][key]
        model.load_state_dict(checkpoint1)
        #model.load_state_dict(checkpoint['model_state_dict'])
        if dataset_name == 'vindr_mammo':  
            valid_dataset = create_valid_dataset(
                VALID_DIR_IMAGES, 
                VALID_DIR_LABELS, 
                IMAGE_SIZE, 
                CLASSES
            )
        if dataset_name == 'DDSM':  
            valid_dataset = create_test_dataset_DDSM(
                VALID_DIR_IMAGES, 
                VALID_DIR_LABELS, 
                IMAGE_SIZE, 
                CLASSES,
                DATA_DIR
            )
    model.to(DEVICE).eval()
    
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)

 

    stats = evaluate(
        model, 
        valid_loader, 
        device=DEVICE,
        num_classes = NUM_CLASSES,
        classes=CLASSES,
    )

    print('\n')
    pprint(stats)