import torch
import cv2
import numpy as np
import os
import glob as glob
import random
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
    
    T.RandomHorizontalFlip(p=0.2),
    T.RandomVerticalFlip(p=0.2),
    T.Scale_box(),
    T.Gaussian_noise(),
    T.RandomResize([size]),
    T.ToTensor(),
])

#train_transform = None
def valid_transform(size):
    return T.Compose([
    T.RandomResize([size]),
    T.ToTensor(),
])

#valid_transfor= None
# Prepare the final datasets and data loaders.
def create_train_dataset_DDSM(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
    data_dir,
):
    train_dataset = DDSMDataset2(
        train_dir_images, 
        train_dir_labels,
        img_size, 
        classes, 
        train_transform(img_size),
        mode='train',
        data_dir= data_dir 
    )
    return train_dataset
def create_valid_dataset_DDSM(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
    data_dir,
):
    valid_dataset = DDSMDataset2(
        train_dir_images, 
        train_dir_labels,
        img_size, 
        classes, 
        valid_transform(img_size),
        mode='val',
        data_dir= data_dir 
    )
    return valid_dataset
def create_test_dataset_DDSM(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
    data_dir,
):
    valid_dataset = DDSMDataset2(
        train_dir_images, 
        train_dir_labels,
        img_size, 
        classes, 
        valid_transform(img_size),
        mode='test',
        data_dir= data_dir 
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

def read_tif(path):
    data = Image.open(path)
    data = np.array(data)
        
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






class DDSMDataset2(Dataset):
    def __init__(
        self, 
        images_path, 
        csv_path,
        img_size, 
        classes, 
        transforms=None, 
        mode='train', 
        data_dir='../'
    ):
        self.transforms = transforms
        self.images_path = images_path
        self.finding_path = csv_path+'/ddsm_description_cases.csv'
        self.img_path = csv_path+f'/data2.csv'
        self.img_size = img_size
        self.classes = classes
        self.mode = mode
        self.all_image_paths = []
        self.data_dir= data_dir
        self.create_anno()
        print(self.annos)

    def create_anno(self):
        finding = pd.read_csv(self.finding_path)
        finding['cases']= finding['patient_id'].apply(lambda s: s.replace('-','_'))
        image_id = pd.read_csv(self.img_path)



        self.image_id= image_id[image_id['split']== self.mode].reset_index()
        
        if self.mode == 'train':
            image_id_mass = (self.image_id['cases']).apply(lambda i: i in set(finding['cases']))
            self.image_id = self.image_id[image_id_mass].reset_index()
        self.annos = finding


            
            
    def load_image_and_labels(self, index):
        image_name = self.image_id['image_id'][index]
        study_id= self.image_id['cases'][index]
        path= self.image_id['path'][index]
        lat = self.image_id['lat'][index]
        # Read the image.
        anno =self.annos[self.annos['image_id']== image_name].reset_index()


        image = read_tif(self.data_dir+path)
        # Convert BGR to RGB color format.
        # Capture the corresponding XML file for getting the annotations.
        
        #print(anno)
        boxes = []
        orig_boxes = []
        labels = []
        image_width = image.shape[1]
        image_height = image.shape[0]
                
        # Box coordinates for xml files are extracted and corrected for image size given.
        for i in range(len(anno)):
            cate = anno['abnormality_type'][i]
            if cate in self.classes:
                labels.append(self.classes.index(cate))
            else:
                continue
                
            # xmin = left corner x-coordinates
            xmin = anno['x_lo'][i]
            # xmax = right corner x-coordinates
            xmax = anno['x_hi'][i]
            # ymin = left corner y-coordinates
            ymin = anno['y_lo'][i]
            # ymax = right corner y-coordinates
            ymax = anno['y_hi'][i]

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
        if lat =='LEFT':
            image, target = Flip(img= image, target = target)
        image, target = self.transforms(image = image, target = target)
        # Fix to enable training without target bounding boxes,
        # see https://discuss.pytorch.org/t/fasterrcnn-images-with-no-objects-present-cause-an-error/117974/4
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
        #debug
        #print(target)
        # if target['boxes'].shape[0]>0:
        #     print(lat)
        #     print(idx)
        #     print(self.image_id['path'][idx])
        #     xmin, ymin, xmax, ymax = target['boxes'][0]
        #     img=image.permute(1,2,0).numpy().copy()
        #     print(img.shape)
        #     img =cv2.rectangle(img = (img*255).astype(np.uint8), pt1= (int(xmin), int(ymin)), pt2= (int(xmax), int(ymax)),color = (255,0,0),thickness= 4)
            
        #     plt.imsave(f'test{idx}.png',img.astype(np.uint8))
        # print(image.shape)
        return image, target


    def __len__(self):
        return len(self.image_id['image_id'])

def create_train_dataset_DDSM_multi(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
    data_dir,
):
    train_dataset = TwoviewDDSMDataset1(
        train_dir_images, 
        train_dir_labels,
        img_size, 
        classes, 
        train_transform(img_size),
        mode='train',
        data_dir= data_dir 
    )
    return train_dataset
def create_valid_dataset_DDSM_multi(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
    data_dir,
):
    valid_dataset = TwoviewDDSMDataset1(
        train_dir_images, 
        train_dir_labels,
        img_size, 
        classes, 
        valid_transform(img_size),
        mode='val',
        data_dir= data_dir 
    )
    return valid_dataset
def create_test_dataset_DDSM_multi(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
    data_dir,
):
    valid_dataset = TwoviewDDSMDataset1(
        train_dir_images, 
        train_dir_labels,
        img_size, 
        classes, 
        valid_transform(img_size),
        mode='test',
        data_dir= data_dir 
    )
    return valid_dataset








class TwoviewDDSMDataset1(Dataset):
    def __init__(
        self, 
        images_path, 
        csv_path,
        img_size, 
        classes, 
        transforms=None, 
        mode='train', 
        data_dir='../'
    ):
        self.transforms = transforms
        self.images_path = images_path
        self.finding_path = csv_path+'/ddsm_description_cases.csv'
        self.img_path = csv_path+f'/data2.csv'
        self.img_size = img_size
        self.classes = classes
        self.mode = mode
        self.all_image_paths = []
        self.log_annot_issue_x = True
        self.log_annot_issue_y = True
        self.data_dir=data_dir
        self.create_anno()

    def create_anno(self):
        finding = pd.read_csv(self.finding_path)
        finding['cases']= finding['patient_id'].apply(lambda s: s.replace('-','_'))
        image_id = pd.read_csv(self.img_path)
        self.image_id= image_id[image_id['split']== self.mode].reset_index()
        if self.mode == 'train':
            image_id_mass = (self.image_id['cases']).apply(lambda i: i in set(finding['cases']))
            self.image_id = self.image_id[image_id_mass].reset_index()
        self.study_id = self.image_id['cases'].unique()
        self.annos = finding
        print(len(self.study_id))


    def load_image_and_labels(self, index,view = 'CC' ):
        if index %2 ==0:
            lat = 'RIGHT'
        else:
            lat = 'LEFT'

        study_id= self.study_id[int(index/2)]
        image_name = self.image_id[self.image_id['cases']== study_id]
        image_name= image_name[image_name['view']== view]
        image_name = image_name[image_name['lat']== lat]['image_id'].values[0]
        path= self.image_id[self.image_id['image_id']== image_name]['path'].values[0]
        # Read the image

        image = read_tif(self.data_dir+path)

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
            # cate = anno['pathology'][i]
            # if cate in self.classes:
            #     labels.append(self.classes.index(cate))
            # elif cate == 'BENIGN_WITHOUT_CALLBACK':
            #     labels.append(self.classes.index('BENIGN'))
            # else:
            #     continue
            cate = anno['abnormality_type'][i]
            if cate in self.classes:
                labels.append(self.classes.index(cate))
            else:
                continue
            # xmin = left corner x-coordinates
            xmin = anno['x_lo'][i]
            # xmax = right corner x-coordinates
            xmax = anno['x_hi'][i]
            # ymin = left corner y-coordinates
            ymin = anno['y_lo'][i]
            # ymax = right corner y-coordinates
            ymax = anno['y_hi'][i]

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
        if lat == 'LEFT':
            image, target= Flip(img= image, target = target)
        image, target = self.transforms(image = image, target = target)
        # Fix to enable training without target bounding boxes,
        # see https://discuss.pytorch.org/t/fasterrcnn-images-with-no-objects-present-cause-an-error/117974/4
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        #debug
        # if target['boxes'].shape[0]>0:
        #     print(int(idx/2))
        #     print(lat)
        #     print(idx)
        #     print(view)
        #     print(self.image_id['path'][idx])
        #     xmin, ymin, xmax, ymax = target['boxes'][0]
        #     img=image.permute(1,2,0).numpy().copy()
        #     print(img.shape)
        #     img =cv2.rectangle(img = (img*255).astype(np.uint8), pt1= (int(xmin), int(ymin)), pt2= (int(xmax), int(ymax)),color = (255,0,0),thickness= 4)
            
        #     plt.imsave(f'test{idx}.png',img.astype(np.uint8))
        return image, target
    def __getitem__(self, idx):
        image_CC, target_CC = self.getitem_view(idx, 'CC')
        image_MLO, target_MLO = self.getitem_view(idx, 'MLO')
        return image_CC, image_MLO, target_CC, target_MLO

    def __len__(self):
        return len(self.study_id)*2
    
    
    
    

















