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
            target_cat=[]
            pred_cat= []
            for j, boxes in enumerate(targets[i]['boxes']):
                
                if targets[i]['labels'][j] == cat:
                    boxes1 = boxes.clone()
                    boxes1[2:] -= boxes1[:2]
                    target_cat.append(boxes1)
            for j, boxes in enumerate(preds[i]['boxes']):
                #print(i, preds[i]['labels'][j], preds[i]['boxes'][j])
                if preds[i]['labels'][j] == cat:
                    boxes1 = boxes.clone()
                    boxes1[2:] -= boxes1[:2]
                    pred_cat.append(boxes1)
            n_gt = len(target_cat)
            n_pr = len(pred_cat)
            #print(i, cat, n_gt, n_pr)
            #print(targets[i]['boxes'])
            #print('pred',preds[i]['boxes'])
            if n_gt == 0:
                if n_pr == 0:
                    continue
                stats[cat]['NL'] += n_pr
                
            else:
                cost_matrix = np.ones((n_gt, n_pr)) * 1e6

                for gt_ind, gt_box in enumerate(target_cat):
                    for pr_ind, pr_box in enumerate(pred_cat):
                        iou_score = get_iou_score(
                            gt_box,
                            pr_box,
                        )
                        #print('iou_score', iou_score)
                        if iou_score > iou_thres:
                            # cost_matrix[gt_ind, pr_ind] = iou_score / (
                            #     np.random.uniform(0, 1) / 1e6
                            # )
                            cost_matrix[gt_ind, pr_ind]= iou_score/1e6
                row_ind, col_ind = linear_sum_assignment(
                    cost_matrix,
                )  # Hungarian-matching
                # print('row_ind',row_ind)
                # print('n_pr',n_pr)
                # print('col_ind',col_ind)
                count=0
                for idx, r in enumerate(row_ind):
                    if cost_matrix[row_ind[idx], col_ind[idx]]<1e-6:
                        count+=1
                n_true_positives = count
                n_false_positives = max(n_pr - count, 0)

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
    def __init__(self, num_classes, classes, threshold =[0.5,1.0,2.0,3.0,4.0], iou_thres= 0.2, n_sample_points= 10000,plot_title='FROC curve CC', view = 'CC'):
        self.threshold = threshold 
        self.iou_thres = iou_thres
        self.n_sample_points = n_sample_points
        self.plot_title = plot_title
        self.num_classes = num_classes
        self.classes = classes
        self.view =view
    def compute(self, preds, targets):
        lls_accuracy = {}
        nlls_per_image = {}
        thres_list = np.linspace(0,1e-3,10, endpoint= True)
        thres_list2 = np.linspace(1e-3,1,1000)
        thres_list = np.append(thres_list, thres_list2[1:])
        print(thres_list)
        first = np.ones([self.num_classes],dtype= bool)
        for score_thres in thres_list:
            #print(score_thres)
            preds= update_scores(preds,score_thres)
            stats = init_stats(targets, self.num_classes)
            
            stats = update_stats(stats, preds, targets, self.iou_thres, self.num_classes)
            lls_accuracy, nlls_per_image = calc_scores(
                stats, lls_accuracy,
                nlls_per_image,
            )
            for k in range(1,self.num_classes):
                if nlls_per_image[k][-1] <1 and first[k]:
                    print(k, score_thres)
                    first[k]=False
        #print(lls_accuracy, nlls_per_image)
        if self.plot_title:
            fig, ax = plt.subplots(figsize=[15, 10])
            ax.set_xticks(
                self.threshold, self.threshold, fontsize=20,
            )
        marker = [None,'*--','.--','+--','o--','>--', '<--']
        output= {}
        for category_id in lls_accuracy:
            lls = lls_accuracy[category_id]
            nlls = nlls_per_image[category_id]
            if self.plot_title:
                ax.plot(
                    nlls,
                    lls,
                    marker[category_id],
                    label=self.classes[category_id] ,
                )
            x= []
            y= []
            for i in range(1,len(nlls)):
                if nlls[i]< nlls[i-1]:
                    x.append(nlls[i])
                    y.append(lls[i])
            print(len(x), len(y))
            print(self.classes[category_id], np.interp( self.threshold, x[::-1], y[::-1]))
            output[self.classes[category_id]]= np.interp( self.threshold, x[::-1], y[::-1])
        count=0
        output['avg'] = np.zeros(len(self.threshold))
        for key in output:
            if key!= 'avg':
                count+=1
                output['avg']+= output[key]
        output['avg'] = output['avg']/count   
        print(output)
        ax.legend(loc = 'best', fontsize = 20)
        ax.set_title(self.plot_title, fontsize=40)
        ax.set_xlabel('False positive per image (FPPI)', fontsize=30)
        ax.set_ylabel('Recall', fontsize=30)
        plt.xlim([0,5])
        plt.savefig(f'result_{self.view}.png')
        plt.close(fig)
        return  output['avg']
