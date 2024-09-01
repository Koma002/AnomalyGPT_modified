import os
from typing import Optional, Callable
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import random


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


class_questions = [
    'This is an image for anomaly detection. What is the content of the image?',
    "What's the object in the image?",
    "What's this in the image?",
    "Describe this image.",
    "Take a look at this image and describe what you notice.",
    "Please provide a description of the picture.",
    "Could you describe the contents of this image for me?",
    # "Can you identify the elements present in the image?"
    # "What can you observe in this picture?",
    # "Describe the objects shown in the image.",
    # "Could you list the items visible in this image?",
    # "What do you see in the picture?",
    # "Identify the various components of this image.",
    # "What is depicted in the photograph?",
    # "Provide a rundown of the contents of this image.",
    # "What's the subject matter of this image?",
    # "Enumerate the objects that can be spotted in this image.",
    # "Describe the visual elements within the picture.",
    # "What visual information can you extract from this image?",
    # "What elements compose the scene in the image?",
    # "Please give a verbal depiction of the image.",
    # "From your perspective, what is shown in the image?",
    # "Could you break down the objects present in the picture?",
    # "Summarize the contents of the image in your own words.",
    # "What details can you identify within the image?",
    # "Provide a textual account of the image's contents.",
    # "Based on the image, can you discern any notable features?"
]


single_answers = [
#    'This in the image is {}.',
    # 'What you\'re seeing here is {}.',
    # 'In this image, the featured object is {}.',
    # '{} is visible in this picture.',
    # 'The object captured in the image is {}.',
    # 'The highlighted item is {}.',
    # 'It appears to be {} in the image.',
    # 'You\'re looking at {} in this photograph.',
    # 'This is none other than {}.',
    # 'The image showcases {}.',
    # 'What\'s presented here is {}.',
    # 'The focus is on {} in this image.',
    # '{} is what we have in the image.',
    # 'The photographed subject is {}.',
    'This image contains {}.',
    # 'The visible entity is {}.',
    # 'The image encapsulates {}.',
    # 'The main subject here is {}.',
    # 'The image portrays {}.',
    # 'The item captured is {}.'
]



anomaly_questions = [
    'Are there any anomalies in the image?',
    'Are there any defects in the image?',
    'Is there any defect in the image?',
    'Is there any anomaly in the image?',
    'Do you observe any irregularities in the image?',
    'Are there any discrepancies in the image?',
    'Can you identify any aberrations in the image?',
    'Do you notice any abnormalities in the image?',
    'Are there any inconsistencies in the image?',
    'Is there any deviance in the image?',
    'Are there any anomalies present in the image?',
    'Do you perceive any faults in the image?',
    'Can you spot any atypical elements in the image?',
    'Are there any variations from the norm in the image?',
    'Do you see any irregular occurrences in the image?',
    'Is there any departure from the standard in the image?',
    'Can you detect any nonconformities in the image?',
    'Are there any divergences in the image?',
    'Do you identify any incongruities in the image?',
    'Is there any departure from expectations in the image?',
    'Are there any aberrant features in the image?',
    'Can you pinpoint any anomalies in the image?',
    'Do you discern any atypical aspects in the image?',
    'Are there any unusual elements in the image?'
]


normal_answers = [
    'No, there is no anomaly in the image.',
    'No, there is no defect in the image.',
    # 'No, there are no anomalies in the image.',
    # 'No, there are no defects in the image.',
    # # "No, this is a photo of {} without any anomalies.",
    # # "No, this is a photo of {} without any defects.",
    # 'No, there is no irregularity in the image.',
    # 'No, there is no imperfection in the image.',
    # 'No, there are no abnormalities in the image.',
    # 'No, there are no blemishes in the image.',
    # 'No, this is a photo of {} without any irregularities.',
    # 'No, this is a photo of {} without any imperfections.',
    # 'No, there are no irregularities present in the image.',
    # 'No, there are no flaws in the image.',
    # 'No, there are no anomalies detected in the image.',
    # 'No, there are no defects to be found in the image.',
    # 'No, this is a photo of {} with no irregularities.',
    # 'No, this is a photo of {} with no imperfections.',
    # 'No, the image is free from irregularities.',
    # 'No, the image does not exhibit any flaws.',
    # 'No, there are no abnormalities observed in the image.',
    # 'No, there are no blemishes spotted in the image.',
    # # 'No, this image of {} shows no irregularities.',
    # # 'No, this image of {} displays no imperfections.',
    # 'No, there are no irregularities visible in the image.',
    # 'No, there are no defects evident in the image.'
]


detail_questions = [
    "What's the anomaly?",
    "What's the defect?",
    "What are the anomalies?",
    "What are the defects?",
    "Why you think so?",
    "What's the irregularity?"
    "What's the flaw?",
    "What are the irregularities?",
    "What are the flaws?",
    "Can you identify the anomaly?",
    "Could you point out the defect?",
    "Do you see any anomalies?",
    "Do you notice any defects?",
    "What's considered anomalous?",
    "What's deemed as a defect?",
    "Can you detect any anomalies?",
    "Can you spot any defects?",
    "What constitutes an anomaly?",
    "What falls under the category of defects?",
    "What's regarded as an anomaly?",
    "What's categorized as a defect?",
    "What anomalies are present?",
    "What defects have been identified?",
    "What kind of anomalies are we looking at?",
    "What types of defects are visible?",
]

def get_class_name(name):
    return 'a type {} resin cutting disc'.format(name[-1])



def format_position(position):
    ret = ""
    for i in range(len(position)):
        if i == 0:
            ret += position[i]
        else:
            if i != len(position) - 1:
                ret += ", "
                ret += position[i]
            else:
                ret += " and " + position[i]

    return ret




class SupervisedSelfDataset(Dataset):
    def __init__(self, root_dir: str):
        import json
        disk_files = ['descriptions/image_descriptions_disc1.json', 'descriptions/image_descriptions_disc2.json', 'descriptions/image_descriptions_disc3.json']
        self.image_descriptions = {}
        for i, file in enumerate(disk_files, 1):
            with open(file, 'r') as json_file:
                self.image_descriptions[f'disc{i}'] = json.load(json_file)
        self.root_dir = root_dir
        self.resize = transforms.Resize(
                                (512, 512), interpolation=transforms.InterpolationMode.BICUBIC
                            )
        
        self.norm_transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    # mean=(0.48145466, 0.4578275, 0.40821073),
                                    # std=(0.26862954, 0.26130258, 0.27577711),
                                    mean=(0.5794, 0.5597, 0.4709),
                                    std=(0.2987, 0.2775, 0.3391),
                                ),
                            ]
                        )

        self.paths = []
        self.descriptions = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if ('gt' not in file_path) and (('jpg' in file_path) or ('png' in file_path)) and "train" in file_path:
                    class_name = file_path.split('/')[-4]
                    self.paths.append(file_path)

                    # Retrieve the corresponding description based on the disk number
                    if 'good' not in file_path:
                        if class_name in self.image_descriptions:
                            cur_disc = self.image_descriptions[class_name]
                            self.descriptions.append(cur_disc[file_path.split('/')[-1]])
                        else:
                            continue

                    else:
                        self.descriptions.append([])


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        img_path = self.paths[index]
        img = self.resize(Image.open(img_path).convert('RGB'))
        class_name = img_path.split('/')[-4]
        descriptions = self.descriptions[index]

        centers = []

        if 'good' not in img_path:
            mask_path = img_path.replace('bad', 'gt')
            mask_path = mask_path.replace('.jpg', '.png')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (512, 512))
            centers = find_contours(mask)
            mask = transforms.ToTensor()(mask)
        else:
            mask = torch.zeros((1,512, 512))


        img = self.norm_transform(img)

        
        position = []
        if len(centers) > 0:
            for center in centers:
                center_y = center[0] / 512
                center_x = center[1] / 512

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
            
            position = list(set(position))

        conversation = []


        r = random.randint(0,2)

        if r == 0:
            conversation.append({"from":"human","value":random.choice(class_questions)})
            conversation.append({"from":"gpt","value":random.choice(single_answers).format(get_class_name(class_name))})
        


        conversation.append({"from":"human","value":random.choice(anomaly_questions)})
        if len(centers) == 0:
            conversation.append({"from":"gpt","value":random.choice(normal_answers).format(get_class_name(class_name))})
        if len(centers) == 1:
            abnormal_describe =  "Yes, there is {} in the image, at the {} of the image.".format(random.choice(['an anomaly','a defect']), position[0])
            # Adding description to the response
            description_text = "The anomaly is described as: " + ', '.join(descriptions) + "."
            conversation.append({"from": "gpt", "value": abnormal_describe + " " + description_text})
        elif len(centers) > 1:
            abnormal_describe =  "Yes, there are {} anomalies in the image, they are at the {} of the image.".format(str(len(centers)), format_position(position))
            # Adding description to the response
            description_text = "The anomalies are described as: " + ', '.join(descriptions) + "."
            conversation.append({"from": "gpt", "value": abnormal_describe + " " + description_text})


        if r == 1:
            conversation.append({"from":"human","value":random.choice(class_questions)})

            conversation.append({"from":"gpt","value":random.choice(single_answers).format(get_class_name(class_name))})

            
        print(img_path, conversation)

        return img, conversation, class_name, mask, img_path



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


        return dict(
            images=images,
            texts=texts,
            class_names=class_names,
            masks=masks,
            img_paths=img_paths
        )