"""
Run evaluation on a trained model to get mAP and class wise AP.

USAGE:
python eval.py --config data_configs/voc.yaml --weights outputs/training/fasterrcnn_convnext_small_voc_15e_noaug/best_model.pth --model fasterrcnn_convnext_small
"""
from datasets import (
    create_valid_dataset_multi, create_valid_loader
)
from models_multiview.multiview_detector import create_model
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
        default=(1400, 1700), 
        type=tuple, 
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
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']
    NUM_WORKERS = args['workers']
    DEVICE = args['device']
    BATCH_SIZE = args['batch']

    # Model configurations
    IMAGE_SIZE = args['imgsz']

    # Load the pretrained model
    if args['weights'] is None:
        try:
            model, coco_model = create_model(num_classes=NUM_CLASSES, size= IMAGE_SIZE, coco_model=True)
        except:
            model = create_model(num_classes=NUM_CLASSES, size= IMAGE_SIZE, coco_model=True)


    # Load weights.
    if args['weights'] is not None:
        model = create_model(num_classes=NUM_CLASSES, size= IMAGE_SIZE,norm = args['norm'], coco_model=False)
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        valid_dataset = create_valid_dataset_multi(
            VALID_DIR_IMAGES, 
            VALID_DIR_LABELS, 
            IMAGE_SIZE, 
            CLASSES
        )
    model.to(DEVICE).eval()
    
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)

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
            true_dict_MLO = dict()
            preds_dict_CC = dict()
            preds_dict_MLO = dict()
            
            def pred(true_dict, preds_dict, targets, outputs, images, preds, target):
                for i in range(len(images)):

                    true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
                    true_dict['labels'] = targets[i]['labels'].detach().cpu()
                    preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                    preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
                    preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
                    preds.append(preds_dict)
                    target.append(true_dict)
                #####################################
            pred(true_dict_CC, preds_dict_CC, targets_CC, outputs_CC, images_CC, preds_CC, target_CC)
            pred(true_dict_MLO, preds_dict_MLO, targets_MLO, outputs_MLO, images_MLO, preds_MLO, target_MLO)
            outputs_CC = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs_CC]
            outputs_MLO = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs_MLO]
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        torch.set_num_threads(n_threads)
        #metric = MeanAveragePrecision(class_metrics=args['verbose'], iou_thresholds =[0.2])
        metric = froc.FROC(num_classes, CLASSES)
        #metric.update(preds, target)
        #metric_summary = metric.compute()
        metric_summary_CC = metric.compute(preds_CC,target_CC)
        metric_summary_MLO = metric.compute(preds_MLO, target_MLO)
        return metric_summary_CC, metric_summary_MLO

    stats_CC, stats_MLO = evaluate(
        model, 
        valid_loader, 
        device=DEVICE,
        num_classes = NUM_CLASSES,
        classes=CLASSES,
    )

    print('\n')
    pprint(stats)
    if args['verbose']:
        print('\n')
        pprint(f"Classes: {CLASSES}")
        print('\n')
        print('AP / AR per class')
        empty_string = ''
        if len(CLASSES) > 2: 
            num_hyphens = 73
            print('-'*num_hyphens)
            print(f"|    | Class{empty_string:<16}| AP{empty_string:<18}| AR{empty_string:<18}|")
            print('-'*num_hyphens)
            class_counter = 0
            for i in range(0, len(CLASSES)-1, 1):
                class_counter += 1
                print(f"|{class_counter:<3} | {CLASSES[i+1]:<20} | {np.array(stats['map_per_class'][i]):.3f}{empty_string:<15}| {np.array(stats['mar_100_per_class'][i]):.3f}{empty_string:<15}|")
            print('-'*num_hyphens)
            print(f"|Avg{empty_string:<23} | {np.array(stats['map']):.3f}{empty_string:<15}| {np.array(stats['mar_100']):.3f}{empty_string:<15}|")
        else:
            num_hyphens = 62
            print('-'*num_hyphens)
            print(f"|Class{empty_string:<10} | AP{empty_string:<18}| AR{empty_string:<18}|")
            print('-'*num_hyphens)
            print(f"|{CLASSES[1]:<15} | {np.array(stats['map']):.3f}{empty_string:<15}| {np.array(stats['mar_100']):.3f}{empty_string:<15}|")
            print('-'*num_hyphens)
            print(f"|Avg{empty_string:<12} | {np.array(stats['map']):.3f}{empty_string:<15}| {np.array(stats['mar_100']):.3f}{empty_string:<15}|")