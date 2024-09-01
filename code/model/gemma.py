from header import *
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from .ImageBind import *
from .ImageBind import data
from .AnomalyGPT_models import LinearLayer, PromptLearner, ProjectionLayer
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils.loss import FocalLoss, BinaryDiceLoss
import kornia as K

import torch
from torch.nn.utils import rnn

CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'object',
               'candle', 'cashew', 'chewinggum', 'fryum', 'macaroni', 'pcb', 'pipe fryum','disc1','disc2','disc3']

DISC_NAMES = ['type 1 resin cutting disc', 'type 2 resin cutting disc', 'type 3 resin cutting disc']

prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']

prompt_state = [prompt_normal, prompt_abnormal]
prompt_templates = ['a photo of a {}.', 'a photo of the {}.']
# prompt_templates = [
#                         'a cropped photo of the {}.', 'a cropped photo of a {}.', 'a close-up photo of a {}.', 'a close-up photo of the {}.',
#                         'a bright photo of the {}.', 'a bright photo of a {}.', 'a dark photo of a {}.', 'a dark photo of the {}.',
#                         'a dark photo of the {}.', 'a dark photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a jpeg corrupted photo of the {}.',
#                         'a blurry photo of the {}.', 'a blurry photo of a {}.', 'a photo of a {}.', 'a photo of the {}.',
#                         'a photo of the small {}.', 'a photo of a small {}.', 'a photo of the large {}.', 'a photo of a large {}.',
#                         'a photo of the {} for visual insprction.', 'a photo of a {} for visual insprction.',
#                         'a photo of the {} for anomaly detection.', 'a photo of a {} for anomaly detection.'
#                         ]
objs = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'object',
        'candle', 'cashew', 'chewinggum', 'fryum', 'macaroni', 'pcb', 'pipe fryum', 'macaroni1', 'macaroni2','pcb1', 'pcb2', 'pcb3', 'pcb4', 'capsules', 'disc1','disc2','disc3']

prompt_sentences = {}

for obj in objs:
    prompt_sentence_obj = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(obj) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = data.load_and_transform_text(prompted_sentence, torch.cuda.current_device())
        prompt_sentence_obj.append(prompted_sentence)
    prompt_sentences[obj] = prompt_sentence_obj



def encode_text_with_prompt_ensemble(model, obj, device):

    global prompt_sentences
    normal_sentences = []
    abnormal_sentences = []
    for idx in range(len(obj)):
        sentence = prompt_sentences[obj[idx].replace('_', ' ')]
        normal_sentences.append(sentence[0])
        abnormal_sentences.append(sentence[1])

    normal_sentences = torch.cat(normal_sentences).to(device)
    abnormal_sentences = torch.cat(abnormal_sentences).to(device)

    class_embeddings_normal = model({ModalityType.TEXT: normal_sentences})[ModalityType.TEXT][0]
    class_embeddings_abnormal = model({ModalityType.TEXT: abnormal_sentences})[ModalityType.TEXT][0]
    # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

    class_embeddings_normal = class_embeddings_normal.reshape((len(obj), len(prompt_templates) * len(prompt_normal), 1024))
    class_embeddings_normal = class_embeddings_normal.mean(dim=1, keepdim=True)
    class_embeddings_normal = class_embeddings_normal / class_embeddings_normal.norm(dim=-1, keepdim=True)

    class_embeddings_abnormal = class_embeddings_abnormal.reshape((len(obj), len(prompt_templates) * len(prompt_abnormal), 1024))
    class_embeddings_abnormal = class_embeddings_abnormal.mean(dim=1, keepdim=True)
    class_embeddings_abnormal = class_embeddings_abnormal / class_embeddings_abnormal.norm(dim=-1, keepdim=True)

    text_features = torch.cat([class_embeddings_normal, class_embeddings_abnormal], dim=1)

    return text_features



class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

def build_one_instance(tokenizer, conversation):
    text_list = []
    input_ids, target_ids = [], []
    for i, turn in enumerate(conversation):
        role = turn['from']
        if i == 0: # the first human turn
            assert role == 'human'
            text = f"{turn['value']}<end_of_turn>\n<start_of_turn>model\n"
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100]*len(one_input_id) # do not perform loss regression on human prompt
        else:
            if role == 'human':
                text = f"\n<start_of_turn>user\n{turn['value']}<end_of_turn>\n<start_of_turn>model\n"
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100]*len(one_input_id)
            elif role == 'gpt':
                text = f"{turn['value']}<end_of_turn>\n<eos>"
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception('Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids

def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len):
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:,:max_tgt_len]
    target_ids = target_ids[:,:max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()

def find_first_file_in_directory(directory_path):
    try:
        file_list = os.listdir(directory_path)
        for item in file_list:
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                return item_path
        return None

    except OSError as e:
        print(f"Error while accessing directory: {e}")
        return None


PROMPT_START = '<start_of_turn>user\n <Img>'
#PROMPT_START = '<start_of_turn>user\n'
class OpenGEMMAPEFTModel(nn.Module):

    '''LoRA for Gemma2-2b model'''

    def __init__(self, **args):
        super(OpenGEMMAPEFTModel, self).__init__()
        self.args = args
        imagebind_ckpt_path = args['imagebind_ckpt_path']
        gemma_ckpt_path = args['gemma_ckpt_path']
        max_tgt_len = args['max_tgt_len']
        stage = args['stage']

        print (f'Initializing visual encoder from {imagebind_ckpt_path} ...')

        self.visual_encoder, self.visual_hidden_size = imagebind_model.imagebind_huge(args)
        imagebind_ckpt = torch.load(imagebind_ckpt_path, map_location=torch.device('cpu'))
        self.visual_encoder.load_state_dict(imagebind_ckpt, strict=True)

        self.iter = 0

        self.image_decoder = LinearLayer(1280, 1024, 4)

        self.prompt_learner = PromptLearner(8, 2048)

        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()
        self.loss_bce = BCEWithLogitsLoss(pos_weight=torch.tensor([100]))


        # free vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print ('Visual encoder initialized.')
        # from {vicuna_ckpt_path}
        print (f'Initializing language decoder from {gemma_ckpt_path}...')

        if args['if_load_decoder']:
            delta_ckpt = torch.load(args['decoder_ckpt_path'], map_location=torch.device('cpu'))
            delta_ckpt = {k.replace("image_decoder.", ""): v for k, v in delta_ckpt.items()}
            self.image_decoder.load_state_dict(delta_ckpt, strict=True)
            print("[!] decoder checkpoint initialized")



        if args['if_load_lora']:
            # add the lora module
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=True,
                r=self.args['lora_r'],
                lora_alpha=self.args['lora_alpha'],
                lora_dropout=self.args['lora_dropout'],
                target_modules=['v_proj', 'q_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj', 'k_proj']
            )
            # quantization_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype="float16",
            #     bnb_4bit_use_double_quant =False,
            # )
            self.tokenizer = AutoTokenizer.from_pretrained(args['lora_ckpt_path'])
            self.gemma_model = AutoModelForCausalLM.from_pretrained(
                gemma_ckpt_path, device_map={"": 0}#, quantization_config=quantization_config
            )
            self.gemma_model = PeftModel.from_pretrained(self.gemma_model, model_id=args['lora_ckpt_path'], config=peft_config)
            print('[!]Peft model initialized.')
        else:
            # add the lora module
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.args['lora_r'],
                lora_alpha=self.args['lora_alpha'],
                lora_dropout=self.args['lora_dropout'],
                target_modules=['v_proj', 'q_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj', 'k_proj']
            )
            # quantization_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype="float16",
            #     bnb_4bit_use_double_quant=False,
            # )
            self.tokenizer = AutoTokenizer.from_pretrained(gemma_ckpt_path)
            self.gemma_model = AutoModelForCausalLM.from_pretrained(
                gemma_ckpt_path, device_map={"": 0}#, quantization_config=quantization_config
            )
            self.gemma_model = get_peft_model(self.gemma_model, peft_config)
        self.gemma_model.print_trainable_parameters()

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        print ('Language decoder initialized.')

        self.gemma_proj = ProjectionLayer(self.visual_hidden_size, self.gemma_model.config.hidden_size)

        self.max_tgt_len = max_tgt_len
        self.device = torch.cuda.current_device()


    def rot90_img(self,x,k):
        # k is 0,1,2,3
        degreesarr = [0., 90., 180., 270., 360]
        degrees = torch.tensor(degreesarr[k]).to(self.gemma_model.dtype).to(self.device)
        x = K.geometry.transform.rotate(x, angle = degrees, padding_mode='reflection')
        return x

    def encode_video(self, video_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_video_data(video_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.gemma_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            video_embeds = embeddings[ModalityType.VISION][0] # bsz x 1024
        inputs_gemma = self.gemma_proj(video_embeds).unsqueeze(1) # bsz x 1 x gemma_size
        atts_gemma = torch.ones(inputs_gemma.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_gemma, atts_gemma

    def encode_audio(self, audio_paths):
        inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            audio_embeds = embeddings[ModalityType.AUDIO][0] # bsz x 1024
        inputs_llama = self.llama_proj(audio_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama

    def encode_thermal(self, thermal_paths):
        inputs = {ModalityType.THERMAL: data.load_and_transform_thermal_data(thermal_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['thermal'][0] # bsz x 1024
        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama

    def encode_image(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.gemma_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024
            patch_features = embeddings['vision'][1] # bsz x [1297, 1, 1280]
        patch_tokens = self.image_decoder(patch_features)

        inputs_gemma = self.gemma_proj(image_embeds) # bsz x 1 x gemma_size
        atts_gemma = torch.ones(inputs_gemma.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_gemma, atts_gemma, patch_tokens
    
    def encode_image_for_web_demo(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data_for_web_demo(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.gemma_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
        patch_tokens = self.image_decoder(patch_features) # bsz x h*w x 1024

        inputs_gemma = self.gemma_proj(image_embeds) # bsz x 1 x gemma_size
        atts_gemma = torch.ones(inputs_gemma.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_gemma, atts_gemma, patch_tokens
    
    def encode_image_for_one_shot(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.gemma_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :]

        return patch_features
    
    def encode_image_for_one_shot_from_tensor(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        inputs = {ModalityType.VISION: torch.stack(image_tensors, dim=0).to(self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.gemma_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :]

        return patch_features
    
    def encode_image_for_one_shot_with_aug(self, image_paths):
        image_tensors = data.load_and_transform_vision_data(image_paths, self.device).to(self.gemma_model.dtype)
        B,C,H,W = image_tensors.shape
        # print(B,C,H,W)

        rotated_images = torch.zeros((4, B, C, H, W)).to(self.gemma_model.dtype).to(self.device)


        for j, degree in enumerate([0, 1, 2, 3]):
            rotated_img = self.rot90_img(image_tensors, degree)
            # 存储旋转后的图像
            rotated_images[j] = rotated_img

        image_tensors = rotated_images.transpose(0,1).reshape(B * 4, C, H, W)

        inputs = {ModalityType.VISION: image_tensors}
        # convert into visual dtype
        inputs = {key: inputs[key] for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :].reshape(B,4,256,1280).reshape(B, 4 * 256, 1280)

        return patch_features
    
    def encode_image_from_tensor(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        inputs = {ModalityType.VISION: torch.stack(image_tensors, dim=0).to(self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.gemma_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024
            patch_features = embeddings['vision'][1] # bsz x h*w x 1024
        patch_tokens = self.image_decoder(patch_features)


        inputs_gemma = self.gemma_proj(image_embeds) # bsz x 1 x gemma_size
        atts_gemma = torch.ones(inputs_gemma.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_gemma, atts_gemma, patch_tokens
    
    def encode_image_from_tensor_no_patch(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        inputs = {ModalityType.VISION: torch.stack(image_tensors, dim=0).to(self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.gemma_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024

        inputs_gemma = self.gemma_proj(image_embeds) # bsz x 1 x gemma_size
        atts_gemma = torch.ones(inputs_gemma.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_gemma, atts_gemma



    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask, anomaly_embedding = None):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        input_ids = input_ids.to(self.device) # bsz x s2
        target_ids = target_ids.to(self.device) # bsz x s2
        attention_mask = attention_mask.to(self.device) # bsz x s2

        batch_size = img_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.tokenizer(p_before,
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_before_embeds = self.gemma_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim

        p_middle = '</Img> '
        #p_middle = ' '
        p_middle_tokens = self.tokenizer(p_middle,
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_middle_embeds = self.gemma_model.model.model.embed_tokens(p_middle_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim


        p_after_embeds = self.gemma_model.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.gemma_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim

        if anomaly_embedding != None:
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_middle_embeds, anomaly_embedding, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim

            empty_targets = (
                torch.ones([batch_size, 1+p_before_embeds.size()[1]+img_embeds.shape[1]+p_middle_embeds.size()[1] + anomaly_embedding.size()[1]], # 1 (bos) + s1 + 1 (image vector)
                        dtype=torch.long).to(self.device).fill_(-100)
            )
            targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]
            atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size()[1]+img_embeds.shape[1]+p_middle_embeds.size()[1] + anomaly_embedding.size()[1]], dtype=torch.long).to(self.device) # bsz x (1 + s1 +1)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
            assert attention_mask.size() == targets.size() # bsz x (1 + s1 + 1 + s2)
            return inputs_embeds, targets, attention_mask 
        else:
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_middle_embeds, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
            # create targets
            empty_targets = (
                torch.ones([batch_size, 1+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1]], # 1 (bos) + s1 + 1 (image vector)
                        dtype=torch.long).to(self.device).fill_(-100)  
            ) # bsz x (1 + s1 + 1)
            targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]

            atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1]], dtype=torch.long).to(self.device) # bsz x (1 + s1 +1)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
            assert attention_mask.size() == targets.size() # bsz x (1 + s1 + 1 + s2)
            return inputs_embeds, targets, attention_mask 


    def forward(self, inputs):

        if 'masks' in inputs:

            image_paths = inputs['images']
            img_embeds, _, patch_tokens = self.encode_image_from_tensor(image_paths)
            class_name = inputs['class_names']
            loss_pixel = 0
            feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, class_name, self.device)

            anomaly_maps = []
            original_anomaly_map = []
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ feats_text_tensor.transpose(-2,-1))
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                temp = anomaly_map.permute(0, 2, 1).view(B, C, H, H)
                original_anomaly_map.append(temp)
                anomaly_map = F.interpolate(temp, size=512, mode='bilinear', align_corners=True)
                anomaly_maps.append(anomaly_map)
            original_anomaly_map = torch.cat(original_anomaly_map, dim=1).to(self.device)
            gt = inputs['masks']
            gt = torch.stack(gt, dim=0).to(self.device)
            gt = gt.squeeze().float()
            gt[gt > 0.3], gt[gt <= 0.3] = 1, 0

            for num in range(len(anomaly_maps)):
                #b_loss = self.loss_bce(anomaly_maps[num][0, 1, :, :], gt)
                anomaly_maps[num] = torch.softmax(anomaly_maps[num], dim=1)
                # f_loss = self.loss_focal(anomaly_maps[num], gt)
                # d_loss = self.loss_dice(anomaly_maps[num][:, 1, :, :], gt)
                # loss_pixel = loss_pixel + d_loss + f_loss + b_loss
            # print(torch.max(anomaly_maps[num][0, 1, :, :]))
            # print(torch.max(anomaly_maps[num][1, 1, :, :]))
            for num in range(len(anomaly_maps)):
                anomaly_maps[num] = anomaly_maps[num][:,1,:,:]
            anomaly_map_all = torch.mean(torch.stack(anomaly_maps, dim=0), dim=0).unsqueeze(1)
        
            if random.randint(0,1) == 3 and len(inputs['img_paths']) == len(image_paths):

                normal_paths = []
                for path in inputs['img_paths']:
                    normal_path = path.replace('test', 'train')
                    normal_path = find_first_file_in_directory("/".join(normal_path.split('/')[:-2])+'/good')
                    normal_paths.append(normal_path)

                print(normal_paths)
                query_patch_tokens = self.encode_image_for_one_shot_from_tensor(image_paths)
                normal_patch_tokens = self.encode_image_for_one_shot_with_aug(normal_paths)
                sims = []
                B = len(image_paths)

                for i in range(len(query_patch_tokens)):
                    query_patch_tokens_reshaped = query_patch_tokens[i].view(B,256,1,1280)
                    normal_tokens_reshaped = normal_patch_tokens[i].reshape(B,1,-1,1280)
                    cosine_similarity_matrix = F.cosine_similarity(query_patch_tokens_reshaped, normal_tokens_reshaped, dim=-1)
                    sim_max, _ = torch.max(cosine_similarity_matrix, dim=-1)
                    sims.append(sim_max)

                sim = torch.mean(torch.stack(sims,dim=0), dim=0).reshape(B,1,16,16)
                sim = F.interpolate(sim,size=512, mode='bilinear', align_corners=True)
                anomaly_map_all = 1 - sim # (anomaly_map_all + 1 - sim) / 2
            anomaly_map_prompts = self.prompt_learner(original_anomaly_map)

            # img_embeds = img_embeds + anomaly_map_prompts
            output_texts = inputs['texts']
            input_ids, target_ids, attention_mask = process_batch_instance(self.tokenizer, output_texts, self.max_tgt_len)
            inputs_embeds, targets, attention_mask = self.prompt_wrap(img_embeds, input_ids, target_ids, attention_mask, anomaly_map_prompts)

            outputs = self.gemma_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            # loss_l2 = torch.norm(anomaly_map_prompts / 2 , p=2)
            # loss_l2 = nn.MSELoss()(img_embeds_origin, img_embeds)
            # calculate the token accuarcy
            chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]    # [B, S-1]
            # print(self.tokenizer.decode(chosen_tokens[0], skip_special_tokens=True))
            labels = targets[:, 2:]
            gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)    # [B*S]
            valid_mask = (labels != -100).reshape(-1)
            print(self.tokenizer.decode(chosen_tokens.reshape(-1)[valid_mask], skip_special_tokens=True))
            valid_tokens = gen_acc & valid_mask    # [B*S]
            gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()

            return loss, gen_acc #loss + loss_pixel, gen_acc
        
        else:
            print('!')
            print('!')
            print('!')
            print('!')
            image_paths = inputs['image_paths']
            img_embeds, _, patch_tokens = self.encode_image_from_tensor(image_paths)

            output_texts = inputs['output_texts']


            feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, ['object'] * len(image_paths), self.device)

            anomaly_maps = []
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)
                # print(patch_tokens[layer].shape)
                # anomaly_map = torch.bmm(patch_tokens[layer], feats_text_tensor.transpose(-2,-1))
                anomaly_map = (100.0 * patch_tokens[layer] @ feats_text_tensor.transpose(-2,-1))
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=512, mode='bilinear', align_corners=True)
                # anomaly_map_no_softmax = anomaly_map
                anomaly_map = torch.softmax(anomaly_map, dim=1)
                anomaly_maps.append(anomaly_map)

            for num in range(len(anomaly_maps)):
                anomaly_maps[num] = anomaly_maps[num][:,1,:,:]

            anomaly_map_all = torch.mean(torch.stack(anomaly_maps, dim=0), dim=0).unsqueeze(1)

            anomaly_map_prompts = self.prompt_learner(anomaly_map_all)

            # img_embeds = img_embeds + anomaly_map_prompts

            input_ids, target_ids, attention_mask = process_batch_instance(self.tokenizer, output_texts, self.max_tgt_len)
            inputs_embeds, targets, attention_mask = self.prompt_wrap(img_embeds, input_ids, target_ids, attention_mask, anomaly_map_prompts)

            outputs = self.gemma_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss
            # calculate the token accuarcy
            chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]    # [B, S-1]
            labels = targets[:, 2:]
            gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)    # [B*S]
            valid_mask = (labels != -100).reshape(-1)
            valid_tokens = gen_acc & valid_mask    # [B*S]
            gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
            
            return loss, gen_acc


    def extract_multimodal_feature(self, inputs, web_demo):
        features = []
        if inputs['image_paths']:
            
            # prompt = inputs['prompt']
            c_name = inputs['class']
            # for name in CLASS_NAMES:
            #     if name in prompt:
            #         c_name = name
            #         break
            # for name in DISC_NAMES:
            #     if name in prompt:
            #         c_name = "disc"+name[5]
            #         break
                
            if not web_demo:
                image_embeds, _, patch_tokens = self.encode_image(inputs['image_paths'])
                feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, [c_name], self.device)
            else:
                image_embeds, _, patch_tokens = self.encode_image_for_web_demo(inputs['image_paths'])
                feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, [c_name], self.device)

            anomaly_maps = []
            original_anomaly_map = []
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)
                # print(patch_tokens[layer].shape)
                # anomaly_map = torch.bmm(patch_tokens[layer], feats_text_tensor.transpose(-2,-1))
                anomaly_map = (100.0 * patch_tokens[layer] @ feats_text_tensor.transpose(-2,-1))
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                temp = anomaly_map.permute(0, 2, 1).view(B, 2, H, H)
                original_anomaly_map.append(temp)
                anomaly_map = F.interpolate(temp,size=512, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)
                anomaly_maps.append(anomaly_map[:,1,:,:])
            anomaly_map_ret = torch.mean(torch.stack(anomaly_maps, dim=0), dim=0).unsqueeze(1)
            original_anomaly_map = torch.cat(original_anomaly_map, dim=1).to(self.device)
            # anomaly_map_all = anomaly_map_ret.unsqueeze(1).repeat((1,3,1,1))
            # anomaly_map_feature, _, _ = self.encode_image_from_tensor(anomaly_map_all)
            # image_embeds = anomaly_map_feature + image_embeds
            if inputs['normal_img_paths']:
                query_patch_tokens = self.encode_image_for_one_shot(inputs['image_paths'])
                if 'mvtec' in 'normal_img_paths':
                    normal_patch_tokens = self.encode_image_for_one_shot_with_aug(inputs['normal_img_paths'])
                else:
                    normal_patch_tokens = self.encode_image_for_one_shot(inputs['normal_img_paths'])
                sims = []

                for i in range(len(query_patch_tokens)):
                    query_patch_tokens_reshaped = query_patch_tokens[i].view(256,1,1280)
                    normal_tokens_reshaped = normal_patch_tokens[i].reshape(1,-1,1280)
                    cosine_similarity_matrix = F.cosine_similarity(query_patch_tokens_reshaped, normal_tokens_reshaped, dim=2)
                    sim_max, _ = torch.max(cosine_similarity_matrix, dim=1)
                    sims.append(sim_max)

                sim = torch.mean(torch.stack(sims,dim=0), dim=0).reshape(1,1,16,16)
                sim = F.interpolate(sim,size=512, mode='bilinear', align_corners=True)
                anomaly_map_ret = 1 - sim # (anomaly_map_ret + 1 - sim) / 2
                

            features.append(image_embeds)
        if inputs['audio_paths']:
            audio_embeds, _ = self.encode_audio(inputs['audio_paths'])
            features.append(audio_embeds)
        if inputs['video_paths']:
            video_embeds, _ = self.encode_video(inputs['video_paths'])
            features.append(video_embeds)
        if inputs['thermal_paths']:
            thermal_embeds, _ = self.encode_thermal(inputs['thermal_paths'])
            features.append(thermal_embeds)

        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return feature_embeds, anomaly_map_ret, original_anomaly_map

    def prepare_generation_embedding(self, inputs, web_demo):
        prompt = inputs['prompt']
        # if len(inputs['modality_embeds']) == 1:
        #     feature_embeds = inputs['modality_embeds'][0]
        # else:
        feature_embeds, anomaly_map, original_anomaly_map = self.extract_multimodal_feature(inputs, web_demo)
        # print(anomaly_map.shape)
        inputs['modality_embeds'].append(feature_embeds)

        batch_size = feature_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.tokenizer(p_before,
            return_tensors="pt", add_special_tokens=False).to(self.device)
        p_before_embeds = self.gemma_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim

        p_middle = '</Img> '
        #p_middle = ' '
        p_middle_tokens = self.tokenizer(p_middle,
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_middle_embeds = self.gemma_model.model.model.embed_tokens(p_middle_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim

        # self.prompt_learner.eval()
        anomaly_map_prompts = self.prompt_learner(original_anomaly_map)
        print(prompt)
        text = prompt + '<end_of_turn>\n<start_of_turn>model\n'
        p_after_tokens = self.tokenizer(text, add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_embeds = self.gemma_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.gemma_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim
        # input_prompt = "<bos><start_of_turn>user\n <Img></Img>"+prompt + '<end_of_turn>\n<start_of_turn>model\n'
        # input_id = self.tokenizer(input_prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
        # print(bos_embeds.shape)
        # print(p_before_embeds.shape)
        # print(feature_embeds.shape)
        # print(p_middle_embeds.shape)
        # print(anomaly_map_prompts.shape)
        # print(p_after_embeds.shape)

        # inputs_embeds = torch.cat(
        #     [bos_embeds, p_before_embeds, anomaly_map_prompts, p_middle_embeds, p_after_embeds],
        #     dim=1)  # bsz x (1+s1+1+s2) x embed_dim
        inputs_embeds = torch.cat([bos_embeds, p_before_embeds, feature_embeds, p_middle_embeds, anomaly_map_prompts, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
    
        return inputs_embeds, anomaly_map#, input_id

    def generate(self, inputs, web_demo=False):
        '''
            inputs = {
                'image_paths': optional,
                'audio_paths': optional
                'video_paths': optional
                'thermal_paths': optional
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature
                'modality_embeds': None or torch.tensor
                'modality_cache': save the image cache
            }
        '''
        # self.prompt_learner.eval()
        # self.gemma_model.eval()
        # self.gemma_proj.eval()
        # self.image_decoder.eval()
        # self.gemma_tokenizer.eval()
        # chat = [
        #     {"role": "user", "content": 'hello_world'},
        # ]
        # prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # print(prompt)
        # inputs = torch.tensor(self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt"))
        # inputs_embeds = self.gemma_model.model.model.embed_tokens(inputs.to(self.gemma_model.device))
        # outputs = self.gemma_model.generate(inputs_embeds=inputs_embeds, max_new_tokens=150)
        # outputs = self.tokenizer.decode(outputs[0])
        # print(outputs)
        input_embeds, pixel_output = self.prepare_generation_embedding(inputs, web_demo)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])
        outputs = self.gemma_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            do_sample=True,
            # use_cache=True,
            # stopping_criteria=stopping_criteria,
        )
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text, pixel_output#output_text