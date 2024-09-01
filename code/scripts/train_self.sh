#!/bin/bash

deepspeed --include localhost:0 --master_port 28412 train_self.py \
    --model gemma_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --gemma_ckpt_path ../pretrained_ckpt/gemma_it_ckpt/\
    --lora_ckpt_path ./ckpt/train_disc/gemma_weight/\
    --decoder_ckpt_path ./ckpt/train_self_decoder_sup/pytorch_model.pt\
    --delta_ckpt_path ./ckpt/train_disc/pytorch_model.pt\
    --no-if_load_lora\
    --if_load_decoder\
    --no-if_load_delta\
    --max_tgt_len 1024\
    --save_path  ./ckpt/train_disc/\
    --log_path ./ckpt/train_disc/log_rest/\
    --data_path ../data/self_final_data_sup
