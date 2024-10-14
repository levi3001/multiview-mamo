import torch
import cv2
import numpy as np
import os
import glob as glob
import random
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datasets import read_xray
import utils.transforms as T
import copy
from tqdm import tqdm


Flip = T.RandomHorizontalFlip(p =1)


class Convert_mammo(Dataset):
    def __init__(
        self, 
        images_path,
        images_path_save,
        val_path,
        img_size, 
        transforms=None, 
    ):
        self.transforms = transforms
        self.images_path = images_path
        self.images_path_save = images_path_save
        self.finding_path = val_path+'/finding_annotations1.csv'
        self.new_finding_path = val_path+'/new_finding_annotations.csv'
        self.breast_level_path = val_path+'/breast-level_annotations1.csv'
        self.img_size = img_size
        self.create_anno()
    def create_anno(self):
        self.finding = pd.read_csv(self.finding_path)
        self.breast_level = pd.read_csv(self.breast_level_path)
        self.new_finding = copy.deepcopy(self.finding)
    

            
            
    def load_image_and_labels(self, index):
        image_name = self.breast_level['image_id'][index]
        study_id= self.breast_level['study_id'][index]
        image_path = os.path.join(self.images_path, study_id+'/'+image_name+ '.dicom')
        lat = self.breast_level['laterality'][index]
        # Read the image.
        anno =self.finding[self.finding['image_id']== image_name]
        image_width = self.breast_level['width'][index]
        image_height = self.breast_level['height'][index]   
        image = read_xray(image_path)
        # Convert BGR to RGB color format.
        # Capture the corresponding XML file for getting the annotations.
        
        #print(anno)
        boxes = []
        orig_boxes = []
        labels = []
        boxes_ind = []
        #image_width = image.shape[1]
        #image_height = image.shape[0]
        # Box coordinates for xml files are extracted and corrected for image size given.
        for i in anno.index:
            if eval(anno['finding_categories'][i])[0] == 'No Finding':
                continue
                
            # Map the current object name to `classes` list to get
            # the label index and append to `labels` list.
            # xmin = left corner x-coordinates
            xmin = anno['xmin'][i]
            # xmax = right corner x-coordinates
            xmax = anno['xmax'][i]
            # ymin = left corner y-coordinates
            ymin = anno['ymin'][i]
            # ymax = right corner y-coordinates
            ymax = anno['ymax'][i]

            xmin, ymin, xmax, ymax = self.check_image_and_annotation(
                xmin, 
                ymin, 
                xmax, 
                ymax, 
                image_width, 
                image_height, 
                orig_data=True
            )

            orig_boxes.append([xmin, ymin, xmax, ymax])
            #print('xmin',xmin)
            # Resize the bounding boxes according to the
            # desired `width`, `height`.
            xmin_final = (xmin/image_width)*image.shape[1]
            xmax_final = (xmax/image_width)*image.shape[1]
            ymin_final = (ymin/image_height)*image.shape[0]
            ymax_final = (ymax/image_height)*image.shape[0]

            xmin_final, ymin_final, xmax_final, ymax_final = self.check_image_and_annotation(
                xmin_final, 
                ymin_final, 
                xmax_final, 
                ymax_final, 
                image.shape[1], 
                image.shape[0],
                orig_data=False
            )
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
            boxes_ind.append(i)
            labels.append(1)
        
        # Bounding box to tensor.
        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Area of the bounding boxes.

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # Labels to tensor.
        #print(labels, boxes)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        return image_name, study_id, image, orig_boxes, \
            boxes, boxes_ind, labels, area, iscrowd, (image_width, image_height), lat

    def check_image_and_annotation(
        self, 
        xmin, 
        ymin, 
        xmax, 
        ymax, 
        width, 
        height, 
        orig_data=False
    ):
        """
        Check that all x_max and y_max are not more than the image
        width or height.
        """
        if ymax > height:
            ymax = height
        if xmax > width:
            xmax = width
        if xmax - xmin <= 1.0:
            if orig_data:
                # print(
                    # '\n',
                    # '!!! xmax is equal to xmin in data annotations !!!'
                    # 'Please check data'
                # )
                # print(
                    # 'Increasing xmax by 1 pixel to continue training for now...',
                    # 'THIS WILL ONLY BE LOGGED ONCE',
                    # '\n'
                # )
                self.log_annot_issue_x = False
            xmin = xmin - 1
        if ymax - ymin <= 1.0:
            if orig_data:
                # print(
                #     '\n',
                #     '!!! ymax is equal to ymin in data annotations !!!',
                #     'Please check data'
                # )
                # print(
                #     'Increasing ymax by 1 pixel to continue training for now...',
                #     'THIS WILL ONLY BE LOGGED ONCE',
                #     '\n'
                # )
                self.log_annot_issue_y = False
            ymin = ymin - 1
        return xmin, ymin, xmax, ymax



    def convert(self):
        for idx in tqdm(range(len(self.breast_level['image_id'].index))):
            # Capture the image name and the full image path.
            image_name, study_id, image, orig_boxes, boxes, boxes_ind, \
                labels, area, iscrowd, size, lat = self.load_image_and_labels(
                index=idx, 
            )



            # Prepare the final `target` dictionary.
            image = Image.fromarray(image)
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["area"] = area
            target["iscrowd"] = iscrowd
            image_id = torch.tensor([idx])
            target["image_id"] = image_id
            if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            if lat =='L':
                image, target = Flip(img= image, target = target)

            image, target = self.transforms(image = image, target = target)
            if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            if not os.path.exists(os.path.join(self.images_path_save, study_id)):
                os.makedirs(os.path.join(self.images_path_save, study_id))
            image.save(os.path.join(self.images_path_save, study_id+'/'+image_name+ '.png'),'PNG')
            boxes = target['boxes']
            for i in range(len(boxes_ind)):
                self.new_finding.loc[boxes_ind[i],'xmin'] = boxes[i][0].item() 
                self.new_finding.loc[boxes_ind[i],'ymin'] = boxes[i][1].item()
                self.new_finding.loc[boxes_ind[i],'xmax'] = boxes[i][2].item()
                self.new_finding.loc[boxes_ind[i],'ymax'] = boxes[i][3].item()
                #print(self.new_finding.iloc[boxes_ind[i]])
                # print(boxes)
            #image = sample['image']
            #target['boxes'] = torch.Tensor(sample['bboxes']).to(torch.int64)
            #target = sample['target']
            # Fix to enable training without target bounding boxes,
            # see https://discuss.pytorch.org/t/fasterrcnn-images-with-no-objects-present-cause-an-error/117974/4
            #debug
            #print(target)
            # if target['boxes'].shape[0]>0:
            #     xmin, ymin, xmax, ymax = target['boxes'][0]
            #     img=image.permute(1,2,0).numpy().copy()
            #     print(img.shape)
            #     img =cv2.rectangle(img = (img*255).astype(np.uint8), pt1= (int(xmin), int(ymin)), pt2= (int(xmax), int(ymax)),color = (255,0,0),thickness= 4)

            #     plt.imsave(f'test{idx}.png',img.astype(np.uint8))
            # print(image.shape)


def train_transform(size):
    return T.Compose([
    T.Breast_crop(),
    T.RandomResize([size]),
])
    
    
convert = Convert_mammo(
        images_path = "/shared/VinDr_Mammo/physionet.org/files/vindr-mammo/1.0.0/images/",
        images_path_save = "/shared/VinDr_Mammo/physionet.org/files/vindr-mammo/1.0.0/images_png",
        val_path=  "/shared/VinDr_Mammo/physionet.org/files/vindr-mammo/1.0.0/",
        img_size = (500,1200), 
        transforms= train_transform((500,1200)), 
    )


convert.convert()
convert.new_finding.to_csv(convert.new_finding_path, index= False)