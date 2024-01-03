"""
Run evaluation on a trained model to get mAP and class wise AP.

USAGE:
python eval.py --config data_configs/voc.yaml --weights outputs/training/fasterrcnn_convnext_small_voc_15e_noaug/best_model.pth --model fasterrcnn_convnext_small
"""
from datasets import (
    create_valid_dataset_multi, create_valid_loader, create_valid_dataset_multi_mask
)
from datasets_DDSM import (
    create_test_dataset_DDSM_multi
)
from models_multiview.create_multiview_model import create_model
from torch_utils import utils
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
from tqdm import tqdm

import torch
import argparse
import yaml
import torchvision
import time
import numpy as np
from torch_utils import froc
torch.multiprocessing.set_sharing_strategy('file_system')

@torch.inference_mode()
def evaluate_mask(
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

    target_CC = []
    preds_CC = []
    target_MLO = []
    preds_MLO = []
    counter = 0
    for images_CC, images_MLO, targets_CC, targets_MLO in tqdm(metric_logger.log_every(data_loader, 100, header), total=len(data_loader)):
        counter += 1
        images_CC = list(image_CC.to(device) for image_CC in images_CC)
        images_MLO = list(image_MLO.to(device) for image_MLO in images_MLO)
        #images = torch.stack(images)
        #print(images.shape)
        #images = images.to(device)
        targets_CC = [{k: v.to(device) for k, v in t.items()} for t in targets_CC]
        targets_MLO = [{k: v.to(device) for k, v in t.items()} for t in targets_MLO]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        with torch.no_grad():
            outputs_CC, outputs_MLO= model(images_CC, images_MLO, targets_CC, targets_MLO)
            #print('out',outputs)
            #print('tar',targets)
        #####################################
        true_dict_CC = dict()
        preds_dict_CC = dict()
        
        def pred(true_dict, preds_dict, targets, outputs, images, preds, target):
            for i in range(len(images)):

                true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
                true_dict['labels'] = targets[i]['labels'].detach().cpu()
                preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
                preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
                preds.append(preds_dict.copy())
                target.append(true_dict.copy())
            #####################################
        pred(true_dict_CC, preds_dict_CC, targets_CC, outputs_CC, images_CC, preds_CC, target_CC)
        outputs_CC = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs_CC]
    # gather the stats from all processes
    
    metric_logger.synchronize_between_processes()
    torch.set_num_threads(n_threads)
    #metric = MeanAveragePrecision(class_metrics=args['verbose'], iou_thresholds =[0.2])
    preds = preds_CC  
    target= target_CC 
    # metric1 = froc.FROC(num_classes, CLASSES, threshold=[0.25,0.5,1,2,4], plot_title= 'FROC curve CC',view= 'CC')
    # #metric.update(preds, target)
    # #metric_summary = metric.compute()
    # metric_summary_CC = metric1.compute(preds_CC,target_CC)
    # metric2 = froc.FROC(num_classes, CLASSES, threshold=[0.25,0.5,1,2,4], plot_title= 'FROC curve MLO', view= 'MLO')
    # metric_summary_MLO = metric2.compute(preds_MLO, target_MLO)
    # return metric_summary_CC, metric_summary_MLO
    metric = froc.FROC(num_classes, classes, threshold= [0.25, 0.5, 1,2,4], plot_title= 'FROC curve all',view = 'all')
    metric_summary =metric.compute(preds, target)
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
        '-v', '--verbose',
        action='store_true',
        help='show class-wise mAP'
    )
    parser.add_argument(
        '--norm', 
        default=None,
        help='normalization type'
    )
    
    parser.add_argument(
        '--use-self-attn',
        dest='use_self_attn',
        action='store_true',
    )
        
    args = vars(parser.parse_args())

    # Load the data configurations
    with open(args['data']) as file:
        data_configs = yaml.safe_load(file)

    # Validation settings and constants.
    try: # Use test images if present.
        VALID_DIR_IMAGES = data_configs['TEST_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['TEST_DIR_LABELS']
    except: # Else use the validation images.
        VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
    dataset_name = data_configs['DATASET']
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']
    NUM_WORKERS = args['workers']
    DEVICE = args['device']
    BATCH_SIZE = args['batch']

    # Model configurations
    IMAGE_SIZE = args['imgsz']

    build_model = create_model[args['model']]
    # Load weights.
    if args['weights'] is not None:
        model = build_model(num_classes=NUM_CLASSES, size= IMAGE_SIZE,norm = args['norm'], use_self_attn= args['use_self_attn'], coco_model=False)
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        checkpoint1 = {}
        for key in checkpoint['model_state_dict']:
            if key.split('.')[0] == 'module':
                checkpoint1[key[7:]]= checkpoint['model_state_dict'][key]
            else:
                checkpoint1[key] = checkpoint['model_state_dict'][key]
        model.load_state_dict(checkpoint1)
        if dataset_name == 'vindr_mammo':
            valid_dataset_nonmask = create_valid_dataset_multi_mask(
                VALID_DIR_IMAGES, 
                VALID_DIR_LABELS, 
                IMAGE_SIZE, 
                CLASSES,
                mask= False
            )
            valid_dataset_mask = create_valid_dataset_multi_mask(
                VALID_DIR_IMAGES, 
                VALID_DIR_LABELS, 
                IMAGE_SIZE, 
                CLASSES,
                mask= True
            )
        elif dataset_name == 'DDSM':
            valid_dataset = create_test_dataset_DDSM_multi(
                VALID_DIR_IMAGES, 
                VALID_DIR_LABELS, 
                IMAGE_SIZE, 
                CLASSES
            )
    model.to(DEVICE).eval()
    
    valid_loader_nonmask = create_valid_loader(valid_dataset_nonmask, BATCH_SIZE, NUM_WORKERS)
    valid_loader_mask = create_valid_loader(valid_dataset_mask, BATCH_SIZE, NUM_WORKERS)

    stats = evaluate_mask(
        model, 
        valid_loader_mask, 
        device=DEVICE,
        num_classes = NUM_CLASSES,
        classes=CLASSES,
    )
    stats1 = evaluate_mask(model, 
        valid_loader_nonmask, 
        device=DEVICE,
        num_classes = NUM_CLASSES,
        classes=CLASSES,
    )

    