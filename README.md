# AnomalyGPT_modified


### Running test file

#### 1. Install Environment

Clone the repository and install the required packages:

```
git clone https://github.com/Koma002/AnomalyGPT_modified.git
pip install -r requirements.txt
```


#### 2. Prepare weights:

##### ImageBind
You can download the pre-trained ImageBind model using [this link](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth). After downloading, put the downloaded file (imagebind_huge.pth) in [[pretrained_ckpt/imagebind_ckpt/]](pretrained_ckpt/) directory. 

##### Gemma
You can download the Gemma model weight from [hugginface](https://huggingface.co/google/gemma-2-2b-it). Once downloaded, put it in [[pretrained_ckpt/gemma_it_ckpt/]](./pretrained_ckpt/) directory.

> After this step, the path `pretrained_ckpt` should look like:
>
    .
    ├── gemma_it_ckpt
    │   ├── README.md
    │   ├── config.json
    │   ├── configuration.json
    │   ├── gemma-2b-it.gguf
    │   ├── generation_config.json
    │   ├── model-00001-of-00002.safetensors
    │   ├── model-00002-of-00002.safetensors
    │   ├── model.safetensors.index.json
    │   ├── special_tokens_map.json
    │   ├── tokenizer.json
    │   ├── tokenizer.model
    │   └── tokenizer_config.json
    └── imagebind_ckpt
        ├── empty.txt
        └── imagebind_huge.pth

##### AnomalyGPT_modified:

You can download AnomalyGPT weights from the table below. 

| Setup and Datasets  | Weights Address |
| :----------------: | :----------------: |
|Unsupervised on MVTec|[train_mvtec_aug](https://drive.google.com/file/d/1Fgyij5UX5SWCfK76fKPpRRcismAhhGsb/view?usp=sharing)|
||[train_mvtec_aug_llm](https://drive.google.com/file/d/1U2j95I8CwuFVES5Kz-qVyIP_8ZcKuTn_/view?usp=sharing)|
| Supervised on cutting disc| [train_self_decoder_sup](https://drive.google.com/file/d/1V7H8phv-nr2d07qBbwror3G02roRVmeq/view?usp=sharing)|
|    | [train_disc](https://drive.google.com/file/d/1aTEF0r8RIWMXoAcu_Z-IbMlsFt9Ei6Q-/view?usp=sharing)|


After downloading, put both weights in the [code/ckpt/](./code/ckpt/) with correspondding folder name.

> After this step, the path `code/ckpt` should look like:
> 
    .
    ├── train_mvtec_aug_llm (or train_disc)
    │   ├── gemma_weight
    │   │   ├── README.md
    │   │   ├── adapter_config.json
    │   │   ├── adapter_model.safetensors
    │   │   ├── special_tokens_map.json
    │   │   ├── tokenizer.json
    │   │   ├── tokenizer.model
    │   │   └── tokenizer_config.json
    │   └── pytorch_model.pt
    └── train_mvtec_aug (or train_self_decoder_sup)
        └── pytorch_model.pt

#### 3. Prepare dataset:
You can download MVTec AD dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads). After downloading, unzip the file and put the folder into [data/](./data/).

If you are interested in cutting disc dataset and would like to test with it, please use the link [here](https://drive.google.com/file/d/1Zo0AGBfxn7P22H66GLk5iF9Hs6s-IWDJ/view?usp=sharing) to apply. 
> After this step, the path `data/` should look like:
>    
    .
    └── mvtec
        ├── bottle
        │   ├── ground_truth
        │   ├── test
        │   └── train
        ├── capsule
        └── ...


#### 4. Run test file

Upon completion of previous steps, you can run the test file as:
```bash
cd ./code/
python test_mvtec.py
```
And the result mask image will generate in `result/` path and the metrics will show on screen. Also, you can use `--if_train_data` or `--no-if_train_data` to switch between train data and test data used. For example:
```bash
python test_mvtec.py --if_train_data
```
****





 