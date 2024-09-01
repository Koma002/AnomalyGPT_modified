import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob

from .self_sup_tasks import patch_ex
from .Draem import augment_image


def find_contours(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            centers.append((center_x, center_y))

    return centers


describles = {}
describles['disc1'] = "This is a photo of a type 1 resin cutting disc1 for anomaly detection, which should be round, without any damage, flaw, defect, scratch, hole or broken part. And the label color is yellow with red and black stripe and some chinese characters on it."
describles['disc2'] = "This is a photo of a type 2 resin cutting disc2 for anomaly detection, which should be round, without any damage, flaw, defect, scratch, hole or broken part. And the label color is orange with red stripe and some chinese characters on it."
describles['disc3'] = "This is a photo of a type 3 resin cutting disc3 for anomaly detection, which should be round, without any damage, flaw, defect, scratch, hole or broken part. And the label color is yellow and blue with orange stripe and some chinese characters on it."
train_set = ['disc1','disc2', 'disc3']

class SelfDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        # self.transform = transform
        self.transform = transforms.Resize(
                                (512, 512), interpolation=transforms.InterpolationMode.BICUBIC
                            )
        
        self.norm_transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                # transforms.Normalize(
                                #     mean=(0.5794, 0.5597, 0.4709),
                                #     std=(0.2987, 0.2775, 0.3391),
                                # ),
                            ]
                        )

        self.paths = []
        self.x = []
        self.anomaly_source_idx = []
        anomaly_source_path = "../data/self_final_data/disc1/train"
        #anomaly_source_path = "../data/dtd/images"
        self.anomaly_path = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if "train" in file_path and "good" in file_path and 'jpg' in file:
                    self.paths.append(file_path)
                    self.x.append(self.transform(Image.open(file_path).convert('RGB')))
                    self.anomaly_source_idx.append(torch.randint(0, len(self.anomaly_path), (1,)).item())

        self.prev_idx = np.random.randint(len(self.paths))


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        img_path, x, anomaly_source_idx = self.paths[index], self.x[index], self.anomaly_source_idx[index]
        class_name = img_path.split('/')[-4]

        self_sup_args={'width_bounds_pct': ((0.03, 0.2), (0.03, 0.2)),
                    'intensity_logistic_params': (1/6, 15),
                    'num_patches': 3, #if single_patch else NUM_PATCHES.get(class_name),
                    'min_object_pct': 0.0,
                    'min_overlap_pct': 0.25,
                    'gamma_params':(2, 0.05, 0.03),
                    'resize':False,
                    'shift':True,
                    'same':False,
                    'mode':'mix',
                    'label_mode':'binary',
                    'skip_background': (200, 60)}

        x = np.asarray(x)#.astype(np.float32) / 255.0
        origin = np.copy(x)

        p = self.x[self.prev_idx]
        if self.transform is not None:
            p = self.transform(p)
        p = np.asarray(p)
        x, mask, centers = patch_ex(x, p, **self_sup_args)
        mask = torch.tensor(mask[None, ..., 0]).float()
        self.prev_idx = index

        # x, mask = augment_image(x, self.anomaly_path[anomaly_source_idx])
        # mask = torch.tensor(mask[None, ..., 0]).float()
        # test = (mask * 255.0).type(torch.uint8).numpy()
        # centers = find_contours(test[0,:,:])


        origin = self.norm_transform(origin)
        x = self.norm_transform(x)
   
        if len(centers) > 0:
            position = []
            for center in centers:
                center_x = center[0] / 224
                center_y = center[1] / 224

                if center_x <= 1/3 and center_y <= 1/3:
                    position.append('top left')
                elif center_x <= 1/3 and center_y > 1/3 and center_y <= 2/3:
                    position.append('top')
                elif center_x <= 1/3 and center_y > 2/3:
                    position.append('top right')

                elif center_x <= 2/3 and center_y <= 1/3:
                    position.append('left')
                elif center_x <= 2/3 and center_y > 1/3 and center_y <= 2/3:
                    position.append('center')
                elif center_x <= 2/3 and center_y > 2/3:
                    position.append('right')

                elif center_y <= 1/3:
                    position.append('bottom left')
                elif center_y > 1/3 and center_y <= 2/3:
                    position.append('bottom')
                elif center_y > 2/3:
                    position.append('bottom right')

            conversation_normal = []
            conversation_normal.append({"from":"human","value": describles[class_name] + " Is there any anomaly in the image?"})
            conversation_normal.append({"from":"gpt","value":"No, there is no anomaly in the image."})
            


            conversation_abnormal = []
            conversation_abnormal.append({"from":"human","value": describles[class_name] + " Is there any anomaly in the image?"})


            
            if len(centers) > 1:
                abnormal_describe =  "Yes, there are " + str(len(centers)) + " anomalies in the image, they are at the "
                for i in range(len(centers)):
                    if i == 0:
                        abnormal_describe += position[i]

                    elif i == 1 and position[i] != position[i-1]:
                        if i != len(centers) - 1:
                            abnormal_describe += ", "
                            abnormal_describe += position[i]
                        else:
                            abnormal_describe += " and " + position[i] + " of the image."
                    
                    elif i == 1 and position[i] == position[i-1]:
                        if i == len(centers) - 1:
                            abnormal_describe += " of the image."

            else:
                abnormal_describe = "Yes, there is an anomaly in the image, at the " + position[0] + " of the image."

            conversation_abnormal.append({"from":"gpt","value":abnormal_describe})

        else:
            print("no mask")
            conversation_normal = []
            conversation_normal.append({"from":"human","value":describles[class_name] + " Is there any anomaly in the image?"})
            conversation_normal.append({"from":"gpt","value":"No, there is no anomaly in the image."})

            conversation_abnormal = conversation_normal
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly>0.5:
            return origin, conversation_normal, class_name, torch.zeros_like(mask), img_path
        else:
            return x, conversation_abnormal, class_name, mask, img_path
        #return origin, conversation_normal, x, conversation_abnormal, class_name, mask, img_path



    def collate(self, instances):

        images = []
        texts = []
        class_names = []
        masks = []
        img_paths = []
        for instance in instances:
            images.append(instance[0])
            texts.append(instance[1])
            class_names.append(instance[2])
            masks.append(instance[3])
            img_paths.append(instance[4])
        # for instance in instances:
        #     images.append(instance[0])
        #     texts.append(instance[1])
        #     class_names.append(instance[4])
        #     masks.append(torch.zeros_like(instance[5]))
        #     img_paths.append(instance[6])
        #
        #     images.append(instance[2])
        #     texts.append(instance[3])
        #     class_names.append(instance[4])
        #     masks.append(instance[5])
        #     img_paths.append(instance[6])

        return dict(
            images=images,
            texts=texts,
            class_names=class_names,
            masks=masks,
            img_paths=img_paths
        )