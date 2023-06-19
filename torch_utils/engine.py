import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
from torch_utils import utils
from torch_utils.coco_eval import CocoEvaluator
from torch_utils.coco_utils import get_coco_api_from_dataset_multi,  get_coco_api_from_dataset
from utils.general import save_validation_results
import numpy as np
def train_one_epoch(
    model, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    train_loss_hist,
    print_freq, 
    scaler=None,
    scheduler=None
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # List to store batch losses.
    batch_loss_list = []
    batch_loss_cls_list = []
    batch_loss_box_reg_list = []
    batch_loss_objectness_list = []
    batch_loss_rpn_list = []

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    step_counter = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        #print('train', type(image))
        #print(images.shape)
        step_counter += 1
        #images = images.to(device)
        images = list(image.to(device) for image in images)
        #images = torch.stack(images)
        #print(images.shape)
        #images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        batch_loss_list.append(loss_value)
        batch_loss_cls_list.append(loss_dict_reduced['loss_classifier'].detach().cpu())
        batch_loss_box_reg_list.append(loss_dict_reduced['loss_box_reg'].detach().cpu())
        batch_loss_objectness_list.append(loss_dict_reduced['loss_objectness'].detach().cpu())
        batch_loss_rpn_list.append(loss_dict_reduced['loss_rpn_box_reg'].detach().cpu())
        train_loss_hist.send(loss_value)

        if scheduler is not None:
            scheduler.step(epoch + (step_counter/len(data_loader)))

    return (
        metric_logger, 
        batch_loss_list, 
        batch_loss_cls_list, 
        batch_loss_box_reg_list, 
        batch_loss_objectness_list, 
        batch_loss_rpn_list
    )


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(
    model, 
    data_loader, 
    coco,
    device, 
    save_valid_preds=False,
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

    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    counter = 0
    for images, targets in metric_logger.log_every(data_loader, 500, header):
        counter += 1
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        if save_valid_preds and counter == 1:
            # The validation prediction image which is saved to disk
            # is returned here which is again returned at the end of the
            # function for WandB logging.
            val_saved_image = save_validation_results(
                images, outputs, counter, out_dir, classes, colors
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    stats = coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return stats, val_saved_image


def train_one_epoch_multi(
    model, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    train_loss_hist,
    print_freq, 
    scaler=None,
    scheduler=None
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # List to store batch losses.
    batch_loss_list = []
    batch_loss_cls_list = []
    batch_loss_box_reg_list = []
    batch_loss_objectness_list = []
    batch_loss_rpn_list = []

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    step_counter = 0
    for images_CC, images_MLO, targets_CC, targets_MLO in metric_logger.log_every(data_loader, print_freq, header):
        #print('train', type(image))
        #print(images.shape)
        step_counter += 1
        #images = images.to(device)
        images_CC = list(image_CC.to(device) for image_CC in images_CC)
        images_MLO = list(image_MLO.to(device) for image_MLO in images_MLO)
        #images = torch.stack(images)
        #print(images.shape)
        #images = images.to(device)
        targets_CC = [{k: v.to(device) for k, v in t.items()} for t in targets_CC]
        targets_MLO = [{k: v.to(device) for k, v in t.items()} for t in targets_MLO]


        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict_CC, loss_dict_MLO = model(images_CC, images_MLO, targets_CC, targets_MLO)
            losses_CC = sum(loss for loss in loss_dict_CC.values())
            losses_MLO = sum(loss for loss in loss_dict_MLO.values())
            losses = losses_CC + losses_MLO
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced_CC = utils.reduce_dict(loss_dict_CC)
        loss_dict_reduced_MLO = utils.reduce_dict(loss_dict_MLO)
        loss_dict_reduced = {}
        for key in loss_dict_reduced_CC:
            loss_dict_reduced[key] = loss_dict_reduced_CC[key]+ loss_dict_reduced_MLO[key]
        
        losses_reduced = sum(loss for loss in loss_dict_reduced_CC.values()) + sum(loss for loss in loss_dict_reduced_MLO.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced_CC, loss_dict_reduced_MLO)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()
        for n, p in model.named_parameters():
            if p.grad is None:
                print(f'{n} has no grad')
        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        batch_loss_list.append(loss_value)
        batch_loss_cls_list.append(loss_dict_reduced['loss_classifier'].detach().cpu())
        batch_loss_box_reg_list.append(loss_dict_reduced['loss_box_reg'].detach().cpu())
        batch_loss_objectness_list.append(loss_dict_reduced['loss_objectness'].detach().cpu())
        batch_loss_rpn_list.append(loss_dict_reduced['loss_rpn_box_reg'].detach().cpu())
        train_loss_hist.send(loss_value)

        if scheduler is not None:
            scheduler.step(epoch + (step_counter/len(data_loader)))

    return (
        metric_logger, 
        batch_loss_list, 
        batch_loss_cls_list, 
        batch_loss_box_reg_list, 
        batch_loss_objectness_list, 
        batch_loss_rpn_list
    )
    
    
@torch.inference_mode()
def evaluate_multi(
    model, 
    data_loader, 
    coco,
    device, 
    save_valid_preds=False,
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

    #coco_CC, coco_MLO = get_coco_api_from_dataset_multi(data_loader.dataset)
    
    iou_types = _get_iou_types(model)
    # coco_evaluator_CC = CocoEvaluator(coco_CC, iou_types)
    # coco_evaluator_MLO = CocoEvaluator(coco_MLO, iou_types)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    counter = 0
    for images_CC, images_MLO, targets_CC, targets_MLO in metric_logger.log_every(data_loader, 100, header):
        counter += 1
        images_CC = list(img.to(device) for img in images_CC)
        images_MLO = list(img.to(device) for img in images_MLO)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs_CC, outputs_MLO = model(images_CC, images_MLO)

        outputs_CC = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs_CC]
        outputs_MLO = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs_MLO]
        model_time = time.time() - model_time

        res_CC = {target["image_id"].item(): output for target, output in zip(targets_CC, outputs_CC)}
        res_MLO = {target["image_id"].item(): output for target, output in zip(targets_MLO, outputs_MLO)}
        res = res_CC.copy()
        res.update(res_MLO)
        evaluator_time = time.time()
        # coco_evaluator_CC.update(res_CC)
        # coco_evaluator_MLO.update(res_MLO)
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)



    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # coco_evaluator_CC.synchronize_between_processes()
    # coco_evaluator_MLO.synchronize_between_processes()
    # # accumulate predictions from all images
    # coco_evaluator_CC.accumulate()
    # stats_CC = coco_evaluator_CC.summarize()
    # coco_evaluator_MLO.accumulate()
    # stats_MLO = coco_evaluator_MLO.summarize()
    # torch.set_num_threads(n_threads)
    # return stats_CC, stats_MLO
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    stats = coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return stats
