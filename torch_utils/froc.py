import numpy as np


from tqdm.auto  import tqdm
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


def update_scores(preds: list, score_thres: float) -> list:
    preds1 = []
    for _, pred in enumerate(preds):
        pred1= {
            'boxes':[],
            'scores' : [],
            'labels' :[]
        }
        for i, score in enumerate(pred['scores']):
            if score > score_thres:
                pred1['scores'].append(pred['scores'][i])
                pred1['boxes'].append(pred['boxes'][i])
                pred1['labels'].append(pred['labels'][i])
        preds1.append(pred1)
    return preds1

def get_overlap(gt_box: list, pr_box: list) -> float:
    """Intersection score between GT and prediction boxes.
    Arguments:
        gt_box {list} -- [x, y, w, h] of ground-truth lesion
        pr_box {list} -- [x, y, w, h] of prediction bounding box
    Returns:
        intersection {float}
    """
    gt_x, gt_y, gt_w, gt_h = gt_box
    pr_x, pr_y, pr_w, pr_h = pr_box

    xA = max(gt_x, pr_x)
    xB = min(gt_x + gt_w, pr_x + pr_w)
    yA = max(gt_y, pr_y)
    yB = min(gt_y + gt_h, pr_y + pr_h)

    return float(max((xB - xA), 0) * max((yB - yA), 0))

def get_iou_score(gt_box: list, pr_box: list) -> float:
    """IoU score between GT and prediction boxes.
    Arguments:
        gt_box {list} -- [x, y, w, h] of ground-truth lesion
        pr_box {list} -- [x, y, w, h] of prediction bounding box
    Returns:
        score {float} -- intersection over union score of the two boxes
    """
    *_, gt_w, gt_h = gt_box
    *_, pr_w, pr_h = pr_box

    intersection = get_overlap(gt_box, pr_box)

    gt_area = gt_w * gt_h
    pr_area = pr_w * pr_h

    return intersection / (gt_area + pr_area - intersection)


def init_stats(targets, num_classes) -> dict:
    """Initializing the statistics before counting leasion
       and non-leasion localiazations.
    Arguments:
        gt {dict} -- Ground truth COCO dataset
        categories {dict} -- Dictionary of categories in the COCO dataset
    Returns:
        stats {dict} -- Statistics to be updated, containing every information
                        necessary to evaluate a single FROC point
    """
    stats = [
        {
            'LL': 0,
            'NL': 0,
            'n_images': 0,
            'n_lesions': 0,
        }
        for i in range(num_classes)
    ]

    for target in targets:
        for label in target['labels']:
            stats[label]['n_lesions']+=1

    for i in range(num_classes):
        stats[i]['n_images']= len(targets)

    return stats


def update_stats(
    stats: dict,
    preds,
    targets,
    iou_thres: float,
    num_classes: int
):
    """Updating statistics as going through images of the dataset.
    Arguments:
        stats {dict} -- FROC statistics
        gt_id_to_annotation {dict} -- Ground-truth image IDs to annotations.
        pr_id_to_annotation {dict} -- Prediction image IDs to annotations.
        categories {dict} -- COCO categories dictionary.
        use_iou {bool} -- Whether or not to use iou thresholding.
        iou_thres {float} -- IoU threshold when using IoU thresholding.
    Returns:
        stats {dict} -- Updated FROC statistics
    """
    for i in range(len(targets)):
        for cat in range(1,num_classes):

            n_gt = len(targets[i]['boxes'])
            n_pr = len(preds[i]['boxes'])
            #print(targets[i]['boxes'])
            #print('pred',preds[i]['boxes'])
            if n_gt == 0:
                if n_pr == 0:
                    continue
                stats[cat]['NL'] += n_pr
                
            else:
                cost_matrix = np.ones((n_gt, n_pr)) * 1e6

                for gt_ind, gt_box in enumerate(targets[i]['boxes']):
                    for pr_ind, pr_box in enumerate(preds[i]['boxes']):
                        iou_score = get_iou_score(
                            gt_box,
                            pr_box,
                        )
                        print('iou_score', iou_score)
                        if iou_score > iou_thres:
                            cost_matrix[gt_ind, pr_ind] = iou_score / (
                                np.random.uniform(0, 1) / 1e6
                            )

                row_ind, col_ind = linear_sum_assignment(
                    cost_matrix,
                )  # Hungarian-matching
                print('row_ind',row_ind)
                print('n_pr',n_pr)
                print('col_ind',col_ind)
                n_true_positives = len(row_ind)
                n_false_positives = max(n_pr - len(col_ind), 0)

                stats[cat]['LL'] += n_true_positives
                stats[cat]['NL'] += n_false_positives

    return stats
def calc_scores(stats, lls_accuracy, nlls_per_image):
    for category_id in range(1,len(stats)):
        if lls_accuracy.get(category_id, None):
            lls_accuracy[category_id].append(
                stats[category_id]['LL'] /
                stats[category_id]['n_lesions'],
            )
        else:
            lls_accuracy[category_id] = []
            lls_accuracy[category_id].append(
                stats[category_id]['LL'] /
                stats[category_id]['n_lesions'],
            )

        if nlls_per_image.get(category_id, None):
            nlls_per_image[category_id].append(
                stats[category_id]['NL'] /
                stats[category_id]['n_images'],
            )
        else:
            nlls_per_image[category_id] = []
            nlls_per_image[category_id].append(
                stats[category_id]['NL'] /
                stats[category_id]['n_images'],
            )

    return lls_accuracy, nlls_per_image
class FROC():
    def __init__(self, num_classes, threshold =[0.5,1.0,2.0,3.0,4.0], iou_thres= 0, n_sample_points= 50,plot_title='FROC curve'):
        self.threshold = threshold 
        self.iou_thres = iou_thres
        self.n_sample_points = n_sample_points
        self.plot_title = plot_title
        self.num_classes = num_classes
    def compute(self, preds, targets):
        lls_accuracy = {}
        nlls_per_image = {}
        for score_thres in tqdm(
                np.linspace(0.0, 1.0, self.n_sample_points, endpoint=False)
        ):
            preds= update_scores(preds,score_thres)
            stats = init_stats(targets, self.num_classes)
            stats = update_stats(stats, preds, targets, self.iou_thres, self.num_classes)
            lls_accuracy, nlls_per_image = calc_scores(
                stats, lls_accuracy,
                nlls_per_image,
            )
        print(lls_accuracy, nlls_per_image)
        if self.plot_title:
            fig, ax = plt.subplots(figsize=[27, 18])
            ins = ax.inset_axes([0.55, 0.05, 0.45, 0.4])
            ins.set_xticks(
                [0.1, 1.0, 2.0, 3.0, 4.0], [
                    0.1, 1.0, 2.0, 3.0, 4.0,
                ], fontsize=30,
            )


        for category_id in lls_accuracy:
            lls = lls_accuracy[category_id]
            nlls = nlls_per_image[category_id]
            if self.plot_title:
                ax.semilogx(
                    nlls,
                    lls,
                    'x--',
                    label='AI ' ,
                )
                ins.plot(
                    nlls,
                    lls,
                    'x--',
                    label='AI ' ,
                )