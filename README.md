# AnomalyGPT_modified


<span id='catelogue'/>

## Catalogue:

* <a href='#test_file'>Running test file</a>




<span id='test_file'/>

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
| Supervised on cutting disc| [train_self_decoder_sup](https://drive.google.com/file/d/1V7H8phv-nr2d07qBbwror3G02roRVmeq/view?usp=sharing)|
|    | [train_disc](https://drive.google.com/file/d/1aTEF0r8RIWMXoAcu_Z-IbMlsFt9Ei6Q-/view?usp=sharing)

After downloading, put both weights in the [code/ckpt/](./code/ckpt/) with correspondding folder name.

> After this step, the path `code/ckpt` should look like:
> 
    .
    ├── train_disc
    │   ├── gemma_weight
    │   │   ├── README.md
    │   │   ├── adapter_config.json
    │   │   ├── adapter_model.safetensors
    │   │   ├── special_tokens_map.json
    │   │   ├── tokenizer.json
    │   │   ├── tokenizer.model
    │   │   └── tokenizer_config.json
    │   └── pytorch_model.pt
    └── train_self_decoder_sup
        └── pytorch_model.pt

#### 3. Prepare dataset:
You can download disc dataset from [here](https://drive.google.com/file/d/1Zo0AGBfxn7P22H66GLk5iF9Hs6s-IWDJ/view?usp=sharing). After downloading, unzip the file and put the folder into [data/](./data/).
> After this step, the path `data/` should look like:
>    
    .
    └── self_final_data_sup
        ├── disc1
        │   ├── test
        │   └── train
        ├── disc2
        │   ├── test
        │   └── train
        └── disc3
            ├── test
            └── train

#### 4. Run test file

Upon completion of previous steps, you can run the test file as:
```bash
cd ./code/
python test_self_gemma.py
```
And the result mask image will generate in `result/` path. Also, you can use `--if_train_data` or `--no-if_train_data` to switch between train data and test data used. For example:
```bash
python test_self_gemma.py --if_train_data
```
****





 