import os
from model.gemma import OpenGEMMAPEFTModel
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import argparse
import json
import cv2

parser = argparse.ArgumentParser("AnomalyGPT", add_help=True)
# paths
parser.add_argument("--few_shot", type=bool, default=False)
parser.add_argument("--k_shot", type=int, default=1)
parser.add_argument("--round", type=int, default=3)
parser.add_argument('--if_train_data', action=argparse.BooleanOptionalAction)


command_args = parser.parse_args()


describles = {}
describles['bottle'] = "This is a photo of a bottle for anomaly detection, which should be round, without any damage, flaw, defect, scratch, hole or broken part."
describles['cable'] = "This is a photo of three cables for anomaly detection, cables cannot be missed or swapped, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['capsule'] = "This is a photo of a capsule for anomaly detection, which should be black and orange, with print '500', without any damage, flaw, defect, scratch, hole or broken part."
describles['carpet'] = "This is a photo of carpet for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['grid'] = "This is a photo of grid for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['hazelnut'] = "This is a photo of a hazelnut for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['leather'] = "This is a photo of leather for anomaly detection, which should be brown and without any damage, flaw, defect, scratch, hole or broken part."
describles['metal_nut'] = "This is a photo of a metal nut for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part, and shouldn't be fliped."
describles['pill'] = "This is a photo of a pill for anomaly detection, which should be white, with print 'FF' and red patterns, without any damage, flaw, defect, scratch, hole or broken part."
describles['screw'] = "This is a photo of a screw for anomaly detection, which tail should be sharp, and without any damage, flaw, defect, scratch, hole or broken part."
describles['tile'] = "This is a photo of tile for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['toothbrush'] = "This is a photo of a toothbrush for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['transistor'] = "This is a photo of a transistor for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['wood'] = "This is a photo of wood for anomaly detection, which should be brown with patterns, without any damage, flaw, defect, scratch, hole or broken part."
describles['zipper'] = "This is a photo of a zipper for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."

FEW_SHOT = command_args.few_shot 
train_data = command_args.if_train_data
mode = 'test'
if train_data:
    mode = "train"
else:
    mode = "test"


# init the model
args = {
    'model': 'gemma_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'gemma_ckpt_path': '../pretrained_ckpt/gemma_it_ckpt/',
    'if_load_lora': True,
    'if_load_decoder':True,
    'lora_ckpt_path': './ckpt/train_mvtec_aug_llm/gemma_weight/',
    'llm_ckpt_path': './ckpt/train_mvtec_aug_llm/pytorch_model.pt',
    'decoder_ckpt_path': './ckpt/train_mvtec_aug/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 6,
    'lora_alpha': 8,
    'lora_dropout': 0.05,
}

def build_directory(path):
    if os.path.exists(path):
        pass
    else: # recursively construct directory
        os.makedirs(path, exist_ok=True)

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

def get_position(center):
    center_y, center_x = center[0] / 512, center[1] / 512

    if center_x <= 1 / 3:
        if center_y <= 1 / 3:
            return 'top left'
        elif center_y <= 2 / 3:
            return 'top'
        else:
            return 'top right'
    elif center_x <= 2 / 3:
        if center_y <= 1 / 3:
            return 'left'
        elif center_y <= 2 / 3:
            return 'center'
        else:
            return 'right'
    else:
        if center_y <= 1 / 3:
            return 'bottom left'
        elif center_y <= 2 / 3:
            return 'bottom'
        else:
            return 'bottom right'

model = OpenGEMMAPEFTModel(**args)

delta_ckpt = torch.load(args['llm_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
delta_ckpt = torch.load(args['decoder_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().cuda()

print(f'[!] init the gemma model over ...')

"""Override Chatbot.postprocess"""

def predict(
    input,
    image_path,
    normal_img_path,
    max_length,
    top_p,
    temperature,
    history,
    modality_cache,
    class_
):
    prompt_text = ''

    for idx, (q, a) in enumerate(history):
        if idx == 0:
            prompt_text += f'{q}<end_of_turn>\n<start_of_turn>model\n {a}\n<end_of_turn>\n'
        else:
            prompt_text += f'<start_of_turn>user\n {q}<end_of_turn>\n<start_of_turn>model\n {a}\n<end_of_turn>'
    if len(history) == 0:
        prompt_text += f'{input}'
    else:
        prompt_text += f' <start_of_turn>user\n {input}'

    response, pixel_output = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [],
        'video_paths': [],
        'thermal_paths': [],
        'normal_img_paths': normal_img_path if normal_img_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache,
        'class': class_
    })

    return response, pixel_output

input = "Is there any anomaly in the image?"
root_dir = '../data/mvtec'

mask_transform = transforms.Compose([
                                transforms.Resize((512, 512)),
                                transforms.ToTensor()
                            ])

CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid','hazelnut', 'leather', 'metal_nut', 'pill', 'screw','tile', 'toothbrush', 'transistor', 'wood', 'zipper']

precision = []
p_auc_list = []
i_auc_list = []

for c_name in CLASS_NAMES:
    normal_img_paths = ["../data/mvtec_anomaly_detection/"+c_name+"/train/good/"+str(command_args.round * 4).zfill(3)+".png", "../data/mvtec_anomaly_detection/"+c_name+"/train/good/"+str(command_args.round * 4 + 1).zfill(3)+".png",
                        "../data/mvtec_anomaly_detection/"+c_name+"/train/good/"+str(command_args.round * 4 + 2).zfill(3)+".png", "../data/mvtec_anomaly_detection/"+c_name+"/train/good/"+str(command_args.round * 4 + 3).zfill(3)+".png"]
    normal_img_paths = normal_img_paths[:command_args.k_shot]
    stats = {'right': 0, 'wrong': 0, 'right_class': 0, 'right_pos': 0, 'total_bad': 0}
    predictions = {'i_pred': [], 'i_label': [], 'p_pred': [], 'p_label': []}
    good_path, bad_path = [f"../result/mvtec_{mode}_all/{c_name}/{cat}/" for cat in ['good', 'bad']]
    for path in [good_path, bad_path]:
        build_directory(path)

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if "test" in file_path and 'png' in file and c_name in file_path:
                normal_img_paths_used = normal_img_paths if FEW_SHOT else []
                resp, anomaly_map = predict(describles[c_name] + '' + input, file_path, normal_img_paths_used, 512,
                                            0.1, 1.0, [], [], c_name)
                is_normal = 'good' in file_path.split('/')[-2]
                anomaly_map = anomaly_map.reshape(512, 512).detach().cpu().numpy()

                if is_normal:
                    img_mask = Image.fromarray(np.zeros((224, 224)), mode='L')
                else:
                    label = file_path.split('/')[-2]
                    mask_path = file_path.replace('test', 'ground_truth')
                    mask_path = mask_path.replace('.png', '_mask.png')
                    img_mask = Image.open(mask_path).convert('L')
                    stats['total_bad'] += 1

                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (512, 512))
                    centers = find_contours(mask)
                    positions = set(get_position(center) for center in centers)
                    # if not any(pos in resp for pos in positions):
                    #     print('!')
                    #     print('!')
                    #     for pos in positions:
                    #         print(pos)
                    #     print('!')
                    stats['right_pos'] += any(pos in resp for pos in positions)

                img_mask = mask_transform(img_mask)
                img_mask[img_mask > 0.1], img_mask[img_mask <= 0.1] = 1, 0
                img_mask = img_mask.squeeze().reshape(512, 512).cpu().numpy()

                predictions['p_label'].append(img_mask)
                predictions['p_pred'].append(anomaly_map)

                predictions['i_label'].append(0 if is_normal else 1)
                predictions['i_pred'].append(anomaly_map.max())

                if ('good' not in file_path and 'Yes' in resp) or ('good' in file_path and 'No' in resp):
                    stats['right'] += 1
                else:
                    stats['wrong'] += 1


                # normalized_map = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map))
                # scaled_map = (normalized_map * 255).astype(np.uint8)
                im = Image.fromarray((np.clip(anomaly_map,0,1)* 255).astype(np.uint8), mode='L')
                if is_normal:
                    img_output_path = good_path
                else:
                    img_output_path = os.path.join(bad_path, label)
                build_directory(img_output_path)
                im.save(os.path.join(img_output_path, file))


    p_auroc = round(roc_auc_score(np.array(predictions['p_label']).ravel(), np.array(predictions['p_pred']).ravel()) * 100, 2)
    i_auroc = round(roc_auc_score(np.array(predictions['i_label']).ravel(), np.array(predictions['i_pred']).ravel()) * 100, 2)

    p_auc_list.append(p_auroc)
    i_auc_list.append(i_auroc)
    precision.append(100 * stats['right'] / (stats['right'] + stats['wrong']))
    print(c_name, 'classification right:', stats['right_class'], 'position right:', stats['right_pos'], 'all anomaly:', stats['total_bad'])
    print(c_name, 'right:', stats['right'], 'wrong:', stats['wrong'])
    print(c_name, "i_AUROC:", i_auroc)
    print(c_name, "p_AUROC:", p_auroc)

print("i_AUROC:",torch.tensor(i_auc_list).mean())
print("p_AUROC:",torch.tensor(p_auc_list).mean())
print("precision:",torch.tensor(precision).mean())