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
from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader
import utils.transforms as T
from PIL import Image


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


Flip = T.RandomHorizontalFlip(p =1)
def train_transform(size):
    return T.Compose([
    T.Breast_crop(),
    T.RandomHorizontalFlip(p=0.2),
    T.RandomVerticalFlip(p=0.2),
    T.Gaussian_noise(),
    T.Scale_box(),
    T.RandomResize([size]),
    T.ToTensor(),
])

#train_transform = None
def valid_transform(size):
    return T.Compose([
    T.Breast_crop(),
    T.RandomResize([size]),
    T.ToTensor(),
])

#valid_transfor= None
# Prepare the final datasets and data loaders.
def create_train_dataset(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
):
    train_dataset = CustomDataset2(
        train_dir_images, 
        train_dir_labels,
        img_size, 
        classes, 
        train_transform(img_size),
        train=True, 
    )
    return train_dataset
def create_valid_dataset(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
):
    valid_dataset = CustomDataset2(
        train_dir_images, 
        train_dir_labels,
        img_size, 
        classes, 
        valid_transform(img_size),
        train=False, 
    )
    return valid_dataset

def create_train_loader(
    train_dataset, batch_size, num_workers=0, batch_sampler=None
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=batch_sampler
    )
    return train_loader

def create_valid_loader(
    valid_dataset, batch_size, num_workers=0, batch_sampler=None
):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=batch_sampler
    )
    return valid_loader

def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    #data = data.astype('float32')
    data = (data*255).astype(np.uint8)
    data = np.repeat(np.expand_dims(data,axis=2),3,2)
        
    return data

def check_intersect(a,b):
    for i in a:
        if i in b:
            return True 
    return False





class CustomDataset2(Dataset):
    def __init__(
        self, 
        images_path, 
        val_path,
        img_size, 
        classes, 
        transforms=None, 
        train=False, 
    ):
        self.transforms = transforms
        self.images_path = images_path
        self.finding_path = val_path+'/finding_annotations.csv'
        self.breast_level_path = val_path+'/breast-level_annotations.csv'
        self.img_size = img_size
        self.classes = classes
        self.train = train
        self.all_image_paths = []
        self.log_annot_issue_x = True
        self.log_annot_issue_y = True
        self.create_anno()
        print(len(self.annos))

    def create_anno(self):
        finding = pd.read_csv(self.finding_path)
        breast_level = pd.read_csv(self.breast_level_path)

        print(finding['finding_categories'].unique())
        finding_mass= (finding['finding_categories']).apply(lambda i: 'No Finding' not in i)
        #finding_mass = (finding['finding_categories']).apply(lambda i : 'Mass' in i or 'Suspicious Calcification' in i)
        #finding_mass = (finding['finding_categories']).apply(lambda i : 'Mass' in i)
        finding = finding[finding_mass]
        if self.train:
            breast_level= breast_level[breast_level['split']== 'training']
            finding =finding[finding['split']== 'training']
        else:
            breast_level= breast_level[breast_level['split']== 'test']
            finding = finding[finding['split']== 'test']
        self.image_id = breast_level[['study_id', 'image_id', 'view_position', 'laterality','height', 'width']].reset_index()
        #print(finding['image_id'])
        
        if self.train:
            image_id_mass = (self.image_id['study_id']).apply(lambda i: i in set(finding['study_id']))
            self.image_id = self.image_id[image_id_mass].reset_index()
        self.annos = finding[['study_id','image_id','height', 'width', 'xmin', 'ymin', 'xmax', 'ymax', 'finding_categories','breast_birads']].reset_index()

            
            
    def load_image_and_labels(self, index):
        image_name = self.image_id['image_id'][index]
        study_id= self.image_id['study_id'][index]
        image_path = os.path.join(self.images_path, study_id+'/'+image_name+ '.dicom')
        lat = self.image_id['laterality'][index]
        # Read the image.
        anno =self.annos[self.annos['image_id']== image_name].reset_index()
        image_width = self.image_id['width'][index]
        image_height = self.image_id['height'][index]   
        image = read_xray(image_path)
        # Convert BGR to RGB color format.
        # Capture the corresponding XML file for getting the annotations.
        
        #print(anno)
        boxes = []
        orig_boxes = []
        labels = []
        #image_width = image.shape[1]
        #image_height = image.shape[0]
                
        # Box coordinates for xml files are extracted and corrected for image size given.
        for i in range(len(anno)):
            # Map the current object name to `classes` list to get
            # the label index and append to `labels` list.
            for cate in eval(anno['finding_categories'][i]):
                if cate in self.classes:
                    labels.append(self.classes.index(cate))
                else:
                    continue
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
        
        # Bounding box to tensor.
        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Area of the bounding boxes.

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)
        #print(labels, boxes)

        return image, orig_boxes, \
            boxes, labels, area, iscrowd, (image_width, image_height), lat

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



    def __getitem__(self, idx):
        # Capture the image name and the full image path.
        image, orig_boxes, boxes, \
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

        #image = sample['image']
        #target['boxes'] = torch.Tensor(sample['bboxes']).to(torch.int64)
        #target = sample['target']
        # Fix to enable training without target bounding boxes,
        # see https://discuss.pytorch.org/t/fasterrcnn-images-with-no-objects-present-cause-an-error/117974/4
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
        #debug
        #print(target)
        # if target['boxes'].shape[0]>0:
        #     xmin, ymin, xmax, ymax = target['boxes'][0]
        #     img=image.permute(1,2,0).numpy().copy()
        #     print(img.shape)
        #     img =cv2.rectangle(img = (img*255).astype(np.uint8), pt1= (int(xmin), int(ymin)), pt2= (int(xmax), int(ymax)),color = (255,0,0),thickness= 4)
            
        #     plt.imsave(f'test{idx}.png',img.astype(np.uint8))
        # print(image.shape)
        return image, target

    def __len__(self):
        return len(self.image_id['image_id'])







def create_train_dataset_multi(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
):
    train_dataset = TwoviewDataset1(
        train_dir_images, 
        train_dir_labels,
        img_size, 
        classes, 
        train_transform(img_size),
        train=True, 
    )
    return train_dataset
def create_valid_dataset_multi(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
):
    valid_dataset = TwoviewDataset1(
        train_dir_images, 
        train_dir_labels,
        img_size, 
        classes, 
        valid_transform(img_size),
        train=False, 

    )
    return valid_dataset






class TwoviewDataset1(Dataset):
    def __init__(
        self, 
        images_path, 
        val_path,
        img_size, 
        classes, 
        transforms=None, 
        train=False, 
    ):
        self.transforms = transforms
        self.images_path = images_path
        self.finding_path = val_path+'/finding_annotations.csv'
        self.breast_level_path = val_path+'/breast-level_annotations.csv'
        self.img_size = img_size
        self.classes = classes
        self.train = train
        self.all_image_paths = []
        self.log_annot_issue_x = True
        self.log_annot_issue_y = True
        self.create_anno()
        print(self.annos)


    def create_anno(self):
        finding = pd.read_csv(self.finding_path)
        breast_level = pd.read_csv(self.breast_level_path)

        print(finding['finding_categories'].unique())
        #finding_mass= (finding['finding_categories']).apply(lambda i: 'No Finding' not in i)
        finding_mass = (finding['finding_categories']).apply(lambda i : 'Mass' in i or 'Suspicious Calcification' in i)
        finding = finding[finding_mass]
        if self.train:
            breast_level= breast_level[breast_level['split']== 'training']
            finding =finding[finding['split']== 'training']
        else:
            breast_level= breast_level[breast_level['split']== 'test']
            finding = finding[finding['split']== 'test']
        self.image_id = breast_level[['study_id', 'image_id', 'view_position', 'laterality']]
        
        if self.train:
            image_id_mass = (self.image_id['study_id']).apply(lambda i: i in set(finding['study_id']))
            self.image_id = self.image_id[image_id_mass].reset_index()
        self.study_id = self.image_id['study_id'].unique()
        print(len(self.study_id))
        self.annos = finding[['study_id','image_id','height', 'width', 'xmin', 'ymin', 'xmax', 'ymax', 'finding_categories','breast_birads']].reset_index()

    def load_image_and_labels(self, index,view = 'CC' ):
        if index %2 ==0:
            lat = 'R'
        else:
            lat = 'L'
        study_id= self.study_id[int(index/2)]
        image_name = self.image_id[self.image_id['study_id']== study_id]
        image_name= image_name[image_name['view_position']== view]
        image_name = image_name[image_name['laterality']== lat]['image_id'].values[0]
        image_path = os.path.join(self.images_path, str(study_id)+'/'+str(image_name)+ '.dicom')

        # Read the image.
        image = read_xray(image_path)
        # Convert BGR to RGB color format.
        
        # Capture the corresponding XML file for getting the annotations.
        anno =self.annos[self.annos['image_id']== image_name].reset_index()
        #print(anno)
        boxes = []
        orig_boxes = []
        labels = []
        image_width = image.shape[1]
        image_height = image.shape[0]
        # Box coordinates for xml files are extracted and corrected for image size given.
        for i in range(len(anno)):
            # Map the current object name to `classes` list to get
            # the label index and append to `labels` list.
            for cate in eval(anno['finding_categories'][i]):
                if cate in self.classes:
                    labels.append(self.classes.index(cate))
                else:
                    continue
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
        
        # Bounding box to tensor.
        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Area of the bounding boxes.

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)
        #print(labels, boxes)
        return image, orig_boxes, \
            boxes, labels, area, iscrowd, (image_width, image_height), lat

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

    def getitem_view(self,idx, view= 'CC'):
        # Capture the image name and the full image path.
        image, orig_boxes, boxes, \
            labels, area, iscrowd, size, lat = self.load_image_and_labels(
                index=idx, view= view
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
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
            
        if lat == 'L':
            image, target= Flip(img= image, target = target)
        image, target = self.transforms(image = image, target = target)
        # Fix to enable training without target bounding boxes,
        # see https://discuss.pytorch.org/t/fasterrcnn-images-with-no-objects-present-cause-an-error/117974/4
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        #debug
        #print(target)
        # if target['boxes'].shape[0]>0:
        #     xmin, ymin, xmax, ymax = target['boxes'][0]
        #     img =cv2.rectangle(img = np.array(image), pt1= (int(xmin), int(ymin)), pt2= (int(xmax), int(ymax)),color = (255,0,0),thickness= 4)
        #     print(img.max())
        #     plt.imsave(f'test{idx}.png',img.astype(np.uint8))
        return image, target
    def __getitem__(self, idx):
        image_CC, target_CC = self.getitem_view(idx, 'CC')
        image_MLO, target_MLO = self.getitem_view(idx, 'MLO')
        return image_CC, image_MLO, target_CC, target_MLO

    def __len__(self):
        return len(self.study_id)*2
    
    
    
    
def create_train_dataset_multi_mask(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
    mask =False,
):
    train_dataset = TwoviewDatasetMask(
        train_dir_images, 
        train_dir_labels,
        img_size, 
        classes, 
        train_transform(img_size),
        train=True,
        mask = mask 
    )
    return train_dataset
def create_valid_dataset_multi_mask(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
    mask= False,
):
    valid_dataset = TwoviewDatasetMask(
        train_dir_images, 
        train_dir_labels,
        img_size, 
        classes, 
        valid_transform(img_size),
        train=False, 
        mask= mask

    )
    return valid_dataset




class TwoviewDatasetMask(Dataset):
    def __init__(
        self, 
        images_path, 
        val_path,
        img_size, 
        classes, 
        transforms=None, 
        train=False, 
        mask = False
    ):
        self.transforms = transforms
        self.images_path = images_path
        self.finding_path = val_path+'/finding_annotations.csv'
        self.breast_level_path = val_path+'/breast-level_annotations.csv'
        self.img_size = img_size
        self.classes = classes
        self.train = train
        self.all_image_paths = []
        self.log_annot_issue_x = True
        self.log_annot_issue_y = True
        self.mask = mask
        self.create_anno()
        print(self.annos)


    def create_anno(self):
        finding = pd.read_csv(self.finding_path)
        breast_level = pd.read_csv(self.breast_level_path)
        #finding_mass= (finding['finding_categories']).apply(lambda i: 'No Finding' not in i)
        finding_mass = (finding['finding_categories']).apply(lambda i : 'Mass' in i or 'Suspicious Calcification' in i)
        finding = finding[finding_mass]
        if self.train:
            breast_level= breast_level[breast_level['split']== 'training']
            finding =finding[finding['split']== 'training']
        else:
            breast_level= breast_level[breast_level['split']== 'test']
            finding = finding[finding['split']== 'test']
    def create_anno(self):
        finding = pd.read_csv(self.finding_path)
        breast_level = pd.read_csv(self.breast_level_path)
        finding_mass= (finding['finding_categories']).apply(lambda i: 'No Finding' not in i)
        finding = finding[finding_mass]
        if self.train:
            breast_level= breast_level[breast_level['split']== 'training']
            finding =finding[finding['split']== 'training']
        else:
            breast_level= breast_level[breast_level['split']== 'test']
            finding = finding[finding['split']== 'test']
        self.image_id = breast_level[['study_id', 'image_id', 'view_position', 'laterality']]
        
        if self.train:
            image_id_mass = (self.image_id['study_id']).apply(lambda i: i in set(finding['study_id']))
            self.image_id = self.image_id[image_id_mass].reset_index()
        self.study_id = self.image_id['study_id'].unique()
        print(len(self.study_id))
        self.annos = finding[['study_id','image_id','height', 'width', 'xmin', 'ymin', 'xmax', 'ymax', 'finding_categories','breast_birads']].reset_index()


    def load_image_and_labels(self, index,view = 'CC' ):
        if index %2 ==0:
            lat = 'R'
        else:
            lat = 'L'
        study_id= self.study_id[int(index/2)]
        image_name = self.image_id[self.image_id['study_id']== study_id]
        image_name= image_name[image_name['view_position']== view]
        image_name = image_name[image_name['laterality']== lat]['image_id'].values[0]
        image_path = os.path.join(self.images_path, str(study_id)+'/'+str(image_name)+ '.dicom')

        # Read the image.
        image = read_xray(image_path)
        # Convert BGR to RGB color format.
        
        # Capture the corresponding XML file for getting the annotations.
        anno =self.annos[self.annos['image_id']== image_name].reset_index()
        #print(anno)
        boxes = []
        orig_boxes = []
        labels = []
        image_width = image.shape[1]
        image_height = image.shape[0]
        # Box coordinates for xml files are extracted and corrected for image size given.
        for i in range(len(anno)):
            # Map the current object name to `classes` list to get
            # the label index and append to `labels` list.
            for cate in eval(anno['finding_categories'][i]):
                if cate in self.classes:
                    labels.append(self.classes.index(cate))
                else:
                    continue
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
                #print(xmin_final)
                #image1 = np.repeat(np.expand_dims(image,2),3, axis=2)
                #plt.imsave(f'infer_{image_name}_{i}.jpg',cv2.rectangle(img =image1,pt1= (int(xmin_final),int(ymin_final)),pt2= (int(xmax_final),int(ymax_final)), color = (1.0,0,0),thickness =2))
                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        # Bounding box to tensor.
        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Area of the bounding boxes.

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)
        #print(labels, boxes)
        return image, orig_boxes, \
            boxes, labels, area, iscrowd, (image_width, image_height), lat

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

    def getitem_view(self,idx, view= 'CC'):
        # Capture the image name and the full image path.
        image, orig_boxes, boxes, \
            labels, area, iscrowd, size, lat = self.load_image_and_labels(
                index=idx, view= view
        )


        # Prepare the final `target` dictionary.
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        if self.mask:
            if view == 'MLO':
                for bbox in target['boxes']:
                    x1, y1, x2, y2 = bbox
                    image[int(y1):int(y2), int(x1):int(x2)]= int(0)
        image = Image.fromarray(image)
        # sample = self.transforms(image=image,
        #                          bboxes=target['boxes'],
        #                          labels=labels)
        if lat == 'L':
            image, target= Flip(img= image, target = target)
        image, target = self.transforms(image = image, target = target)

        
        # Fix to enable training without target bounding boxes,
        # see https://discuss.pytorch.org/t/fasterrcnn-images-with-no-objects-present-cause-an-error/117974/4
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        #print(target)
        # if target['boxes'].shape[0]>0:
        #     xmin, ymin, xmax, ymax = target['boxes'][0]
        #     img = np.array(image).transpose((1,2,0)).copy()
        #     img = (img*255).astype('uint8')
        #     img =cv2.rectangle(img =img, pt1= (int(xmin), int(ymin)), pt2= (int(xmax), int(ymax)),color = (255,0,0),thickness= 4)
        #     print(view)
        #     plt.imsave(f'test{view}{idx}.png',img)
        return image, target
    def __getitem__(self, idx):
        image_CC, target_CC = self.getitem_view(idx, 'CC')
        image_MLO, target_MLO = self.getitem_view(idx, 'MLO')
        return image_CC, image_MLO, target_CC, target_MLO

    def __len__(self):
        return len(self.study_id)*2
def create_train_dataset_multi_mask(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
    mask =False,
):
    train_dataset = TwoviewDatasetMask(
        train_dir_images, 
        train_dir_labels,
        img_size, 
        classes, 
        train_transform(img_size),
        train=True,
        mask = mask 
    )
    return train_dataset
def create_valid_dataset_multi_mask(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
    mask= False,
):
    valid_dataset = TwoviewDatasetMask(
        train_dir_images, 
        train_dir_labels,
        img_size, 
        classes, 
        valid_transform(img_size),
        train=False, 
        mask= mask

    )
    return valid_dataset




class TwoviewDatasetMask(Dataset):
    def __init__(
        self, 
        images_path, 
        val_path,
        img_size, 
        classes, 
        transforms=None, 
        train=False, 
        mask = False
    ):
        self.transforms = transforms
        self.images_path = images_path
        self.finding_path = val_path+'/finding_annotations.csv'
        self.breast_level_path = val_path+'/breast-level_annotations.csv'
        self.img_size = img_size
        self.classes = classes
        self.train = train
        self.all_image_paths = []
        self.log_annot_issue_x = True
        self.log_annot_issue_y = True
        self.mask = mask
        self.create_anno()
        print(self.annos)


    def create_anno(self):
        finding = pd.read_csv(self.finding_path)
        breast_level = pd.read_csv(self.breast_level_path)
        #finding_mass= (finding['finding_categories']).apply(lambda i: 'No Finding' not in i)
        finding_mass = (finding['finding_categories']).apply(lambda i : 'Mass' in i or 'Suspicious Calcification' in i)
        finding = finding[finding_mass]
        if self.train:
            breast_level= breast_level[breast_level['split']== 'training']
            finding =finding[finding['split']== 'training']
        else:
            breast_level= breast_level[breast_level['split']== 'test']
            finding = finding[finding['split']== 'test']
        self.image_id = breast_level[['study_id', 'image_id', 'view_position', 'laterality']]
        
        if self.train:
            image_id_mass = (self.image_id['study_id']).apply(lambda i: i in set(finding['study_id']))
            self.image_id = self.image_id[image_id_mass].reset_index()
        self.study_id = self.image_id['study_id'].unique()
        print(len(self.study_id))
        self.annos = finding[['study_id','image_id','height', 'width', 'xmin', 'ymin', 'xmax', 'ymax', 'finding_categories','breast_birads']].reset_index()


    def load_image_and_labels(self, index,view = 'CC' ):
        if index %2 ==0:
            lat = 'R'
        else:
            lat = 'L'
        study_id= self.study_id[int(index/2)]
        image_name = self.image_id[self.image_id['study_id']== study_id]
        image_name= image_name[image_name['view_position']== view]
        image_name = image_name[image_name['laterality']== lat]['image_id'].values[0]
        image_path = os.path.join(self.images_path, str(study_id)+'/'+str(image_name)+ '.dicom')

        # Read the image.
        image = read_xray(image_path)
        # Convert BGR to RGB color format.
        
        # Capture the corresponding XML file for getting the annotations.
        anno =self.annos[self.annos['image_id']== image_name].reset_index()
        #print(anno)
        boxes = []
        orig_boxes = []
        labels = []
        image_width = image.shape[1]
        image_height = image.shape[0]
        # Box coordinates for xml files are extracted and corrected for image size given.
        for i in range(len(anno)):
            # Map the current object name to `classes` list to get
            # the label index and append to `labels` list.
                # if anno['breast_birads'][i] in ['BI-RADS 3', 'BI-RADS 4', 'BI-RADS 5']:
                #     labels.append(self.classes.index('malignancy'))
                # else:
                #     labels.append(self.classes.index('__background__'))
            for cate in eval(anno['finding_categories'][i]):
                if cate in self.classes:
                    labels.append(self.classes.index(cate))
                else:
                    continue
                    labels.append(self.classes.index('Other'))
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
                #print(xmin_final)
                #image1 = np.repeat(np.expand_dims(image,2),3, axis=2)
                #plt.imsave(f'infer_{image_name}_{i}.jpg',cv2.rectangle(img =image1,pt1= (int(xmin_final),int(ymin_final)),pt2= (int(xmax_final),int(ymax_final)), color = (1.0,0,0),thickness =2))
                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        # Bounding box to tensor.
        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Area of the bounding boxes.

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)
        #print(labels, boxes)
        return image, orig_boxes, \
            boxes, labels, area, iscrowd, (image_width, image_height), lat

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

    def getitem_view(self,idx, view= 'CC'):
        # Capture the image name and the full image path.
        image, orig_boxes, boxes, \
            labels, area, iscrowd, size, lat = self.load_image_and_labels(
                index=idx, view= view
        )

        
        # visualize_mosaic_images(boxes, labels, image_resized, self.classes)

        # Prepare the final `target` dictionary.
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        if self.mask:
            if view == 'MLO':
                for bbox in target['boxes']:
                    x1, y1, x2, y2 = bbox
                    image[int(y1):int(y2), int(x1):int(x2)]= int(0)
        image = Image.fromarray(image)
        # sample = self.transforms(image=image,
        #                          bboxes=target['boxes'],
        #                          labels=labels)
        if lat == 'L':
            image, target= Flip(img= image, target = target)
        image, target = self.transforms(image = image, target = target)

        #image = sample['image']
        #target['boxes'] = torch.Tensor(sample['bboxes']).to(torch.int64)
        #target = sample['target']
        # Fix to enable training without target bounding boxes,
        # see https://discuss.pytorch.org/t/fasterrcnn-images-with-no-objects-present-cause-an-error/117974/4
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        #debug
        #print(target)
        # if target['boxes'].shape[0]>0:
        #     xmin, ymin, xmax, ymax = target['boxes'][0]
        #     img = np.array(image).transpose((1,2,0)).copy()
        #     img = (img*255).astype('uint8')
        #     img =cv2.rectangle(img =img, pt1= (int(xmin), int(ymin)), pt2= (int(xmax), int(ymax)),color = (255,0,0),thickness= 4)
        #     print(view)
        #     plt.imsave(f'test{view}{idx}.png',img)
        return image, target
    def __getitem__(self, idx):
        image_CC, target_CC = self.getitem_view(idx, 'CC')
        image_MLO, target_MLO = self.getitem_view(idx, 'MLO')
        return image_CC, image_MLO, target_CC, target_MLO

    def __len__(self):
        return len(self.study_id)*2



















class TwoviewDataset(Dataset):
    def __init__(
        self, 
        images_path, 
        val_path,
        img_size, 
        classes, 
        transforms=None, 
        train=False, 
    ):
        self.transforms = transforms
        self.images_path = images_path
        self.finding_path = val_path+'/finding_annotations.csv'
        self.breast_level_path = val_path+'/breast-level_annotations.csv'
        self.img_size = img_size
        self.classes = classes
        self.train = train
        self.all_image_paths = []
        self.log_annot_issue_x = True
        self.log_annot_issue_y = True
        self.create_anno()
        print(self.annos)

    def create_anno(self, lat ='R'):
        finding = pd.read_csv(self.finding_path)
        breast_level = pd.read_csv(self.breast_level_path)
        finding= finding[finding['laterality'] == lat ]
        finding_mass= (finding['finding_categories']).apply(lambda i: 'No Finding' not in i)
        finding = finding[finding_mass]
        breast_level = breast_level[breast_level['laterality'] == lat]
        if self.train:
            breast_level= breast_level[breast_level['split']== 'training']
            finding =finding[finding['split']== 'training']
        else:
            breast_level= breast_level[breast_level['split']== 'test']
            finding = finding[finding['split']== 'test']
        self.image_id = breast_level[['study_id', 'image_id', 'view_position', 'laterality']]
        
        if self.train:
            image_id_mass = (self.image_id['study_id']).apply(lambda i: i in set(finding['study_id']))
            self.image_id = self.image_id[image_id_mass].reset_index()
        self.study_id = self.image_id['study_id'].unique()
        print(len(self.study_id))
        self.annos = finding[['study_id','image_id','height', 'width', 'xmin', 'ymin', 'xmax', 'ymax', 'finding_categories','breast_birads']].reset_index()


    def load_image_and_labels(self, index,view = 'CC' ):
        study_id= self.study_id[index]
        image_name = self.image_id[self.image_id['study_id']== study_id]
        image_name= image_name[image_name['view_position']== view]['image_id'].values[0]
        image_path = os.path.join(self.images_path, str(study_id)+'/'+str(image_name)+ '.dicom')

        # Read the image.
        image = read_xray(image_path)
        # Convert BGR to RGB color format.
        
        # Capture the corresponding XML file for getting the annotations.
        anno =self.annos[self.annos['image_id']== image_name].reset_index()
        #print(anno)
        boxes = []
        orig_boxes = []
        labels = []
        image_width = image.shape[1]
        image_height = image.shape[0]
                
        # Box coordinates for xml files are extracted and corrected for image size given.
        for i in range(len(anno)):
            # Map the current object name to `classes` list to get
            # the label index and append to `labels` list.
                # if anno['breast_birads'][i] in ['BI-RADS 3', 'BI-RADS 4', 'BI-RADS 5']:
                #     labels.append(self.classes.index('malignancy'))
                # else:
                #     labels.append(self.classes.index('__background__'))
            for cate in eval(anno['finding_categories'][i]):
                if cate in self.classes:
                    labels.append(self.classes.index(cate))
                else:
                    #continue
                    labels.append(self.classes.index('Other'))
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
                #print(xmin_final)
                #image1 = np.repeat(np.expand_dims(image,2),3, axis=2)
                #plt.imsave(f'infer_{image_name}_{i}.jpg',cv2.rectangle(img =image1,pt1= (int(xmin_final),int(ymin_final)),pt2= (int(xmax_final),int(ymax_final)), color = (1.0,0,0),thickness =2))
                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        # Bounding box to tensor.
        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Area of the bounding boxes.

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)
        #print(labels, boxes)
        return image, orig_boxes, \
            boxes, labels, area, iscrowd, (image_width, image_height)

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

    def getitem_view(self,idx, view= 'CC'):
        # Capture the image name and the full image path.
        image, orig_boxes, boxes, \
            labels, area, iscrowd, dims = self.load_image_and_labels(
                index=idx, view= view
        )

        
        # visualize_mosaic_images(boxes, labels, image_resized, self.classes)

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
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
            
        # sample = self.transforms(image=image,
        #                          bboxes=target['boxes'],
        #                          labels=labels)
        image, target = self.transforms(image = image, target = target)
        #image = sample['image']
        #target['boxes'] = torch.Tensor(sample['bboxes']).to(torch.int64)
        #target = sample['target']
        # Fix to enable training without target bounding boxes,
        # see https://discuss.pytorch.org/t/fasterrcnn-images-with-no-objects-present-cause-an-error/117974/4
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        #print(target)
        # if target['boxes'].shape[0]>0:
        #     xmin, ymin, xmax, ymax = target['boxes'][0]
        #     img =cv2.rectangle(img = np.array(image), pt1= (int(xmin), int(ymin)), pt2= (int(xmax), int(ymax)),color = (255,0,0),thickness= 4)
        #     print(img.max())
        #     plt.imsave(f'test{idx}.png',img.astype(np.uint8))
        return image, target
    def __getitem__(self, idx):
        image_CC, target_CC = self.getitem_view(idx, 'CC')
        image_MLO, target_MLO = self.getitem_view(idx, 'MLO')
        return image_CC, image_MLO, target_CC, target_MLO

    def __len__(self):
        return len(self.study_id)