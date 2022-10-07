# UniTAB: Unifying Text and Box Outputs for Grounded VL Modeling

This repo is forked from [Unitab](https://github.com/microsoft/UniTAB), which support train by [ColossalAI](https://github.com/hpcaitech/ColossalAI)(Colossal-AI: A Unified Deep Learning System for Big Model Era)

## Installation

Clone the repository:
```
git clone https://github.com/microsoft/UniTAB.git
cd UniTAB
```

New conda env:
```
conda create -n unitab python=3.8
conda activate unitab
```

Install packages in ``requirements.txt`` (separately install [numpy](https://pypi.org/project/numpy/) and [pytorch (LTS 1.8.2)](https://pytorch.org/get-started/locally/) if fails):
```
pip install -r requirements.txt
```

### AzCopy
We recommend using the following AzCopy command to download.
AzCopy executable tools can be [downloaded here](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy).

Example command:
```
path/to/azcopy copy <folder-link> <target-address> --resursive"

# For example:
path/to/azcopy copy https://unitab.blob.core.windows.net/data/data <local_path> --recursive
path/to/azcopy copy https://unitab.blob.core.windows.net/data/weights <local_path> --recursive
path/to/azcopy copy https://unitab.blob.core.windows.net/data/annotations <local_path> --recursive
```

## Data

* Download the original Flickr30k image dataset from : [Flickr30K webpage](http://shannon.cs.illinois.edu/DenotationGraph/) and update the `flickr_img_path` to the folder containing the images.
* Download the original Flickr30k entities annotations from: [Flickr30k annotations](https://github.com/BryanPlummer/flickr30k_entities) and update the `flickr_dataset_path` to the folder with annotations.
* Download the gqa images at [GQA images](https://nlp.stanford.edu/data/gqa/images.zip) and update `vg_img_path` to point to the folder containing the images.
* Download COCO images [Coco train2014](http://images.cocodataset.org/zips/train2014.zip). Update the `coco_path` to the folder containing the downloaded images.

Or download the [cached data (~77G)](https://unitab.blob.core.windows.net/data/data) (use AzCopy with the link).

* Download our pre-processed [annotations (~3.7G)](https://unitab.blob.core.windows.net/data/annotations) (use AzCopy with the link, or [zip file](https://unitab.blob.core.windows.net/data/annotations.zip)) and update the `flickr_ann_path`, `gqa_ann_path` and `refexp_ann_path` to this folder with pre-processed annotations.

## Pre-train
The config file for pretraining is ``configs/pretrain.json``. Optionally starting from [MDETR](https://github.com/ashkamath/mdetr/blob/main/.github/pretrain.md) pretrain with ``--load https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth``. [Weights availble here](https://unitab.blob.core.windows.net/data/weights/pretrained_checkpoint.pth).

Example command (ngpu=64):
```
CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/pretrain.json --batch_size 2 --lr_backbone 2e-5 --text_encoder_lr 2e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --do_caption --no_detection --unitab_pretrain --pretrain_seqcrop mixed --ema --output-dir weights/$exp_id --load https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth
```

## Distributed Training by colossalAi
We offer a shell script `colossalai.sh` to support distributed train by colossalai or run the command below

```shell
CUBLAS_WORKSPACE_CONFIG=:4096:8  

torchrun --nproc_per_node=4 --master_port 29505  main.py \
    --dataset_config configs/pretrain_test_flickr_only.json \
    --colossalai_config config.py \
    --batch_size 4 \
    --lr_backbone 2e-5 \
    --text_encoder_lr 2e-5 \
    --lr 1e-4 \
    --num_queries 200 \
    --max_decoding_step 256 \
    --do_caption \
    --no_detection \
    --unitab_pretrain \
    --pretrain_seqcrop mixed \
    --ema \
    --output-dir weights/$exp_id \
    --distributed \
    --load path/to/weights/pretrained_checkpoint.pth \
    --from_colossalai \
```

## Multi-task Finetuning
The config file for pretraining is ``configs/multitask.json``. [Weights availble here](https://unitab.blob.core.windows.net/data/weights/prefinetune_checkpoint.pth).

Example command (ngpu=32):
```
CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/multitask.json --batch_size 2 --lr_backbone 1e-5 --text_encoder_lr 1e-5 --lr 5e-5 --num_queries 200 --max_decoding_step 256 --load weights/pretrained_checkpoint.pth --ema --output-dir weights/$exp_id
```

## Downstream tasks
Optionally, downloading all weights at once (~54G):
```
path/to/azcopy copy https://unitab.blob.core.windows.net/data/weights <local_path> --recursive
```

For model inference, use the input arguments ``--eval --test``. For captioning tests (Flickr grounded captioning, COCO image captioning, VQAv2 visual question answering), the computed captioning metrics displayed is only for reference. For the final number, an output prediction json file will be automatically stored at ``weights/$exp_id/results/pred_dict_$CIDEr.json``. Please follow the official evaluation for [Flickr grounded captioning](https://github.com/facebookresearch/grounded-video-description), [COCO captioning](https://github.com/tylin/coco-caption), and [VQAv2](https://visualqa.org/evaluation.html) evaluation. We will better intergrate the caption evaluations in future versions.

### Grounded captioning
The config file for pretraining is ``configs/flickr_kp.json``. For model inference, use the input arguments ``--eval --test``. 

For the final number, an output prediction json file will be automatically stored at ``weights/$exp_id/results/pred_dict_$CIDEr.json``. Please follow the official evaluation for [Flickr grounded captioning](https://github.com/facebookresearch/grounded-video-description) evaluation. We will better intergrate the caption evaluations in future versions.

Weights: [Separate](https://unitab.blob.core.windows.net/data/weights/separate_flickrcaptionKP_checkpoint.pth), [Pre-finetuning](https://unitab.blob.core.windows.net/data/weights/prefinetune_flickrcaptionKP_checkpoint.pth).

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>CIDEr</th>
            <th>F1_all</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Separate</td>
            <td>65.6</td>
            <td>11.46</td>
        </tr>
        <tr>
            <td>Pre-finetuning</td>
            <td>69.7</td>
            <td>12.95 </td>
        </tr>
    </tbody>
</table>

Example command (ngpu=8):
```
CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/flickr_kp.json --batch_size 2 --lr_backbone 1e-5 --text_encoder_lr 1e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --do_caption --no_detection --ema --output-dir weights/$exp_id --load weights/pretrained_checkpoint.pth

CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/flickr_kp.json --batch_size 2 --lr_backbone 1e-5 --text_encoder_lr 1e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --do_caption --no_detection --ema --output-dir weights/$exp_id --load weights/prefinetune_flickrcaptionKP_checkpoint.pth --eval --test
```

### Referring expression comprehension
The config file for pretraining is ``configs/refcoco/+/g.json``. For model inference, use the input arguments ``--eval --test --test_type testA/testB/test``.

Weights: [Separate](https://unitab.blob.core.windows.net/data/weights/separate_refcoco_checkpoint.pth), [Pre-finetuning](https://unitab.blob.core.windows.net/data/weights/prefinetune_refcoco_checkpoint.pth) (refcoco/refcoco+/refcocog).

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Refcoco</th>
            <th>Refcoco+</th>
            <th>Refcocog</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Separate</td>
            <td>86.32</td>
            <td>78.70</td>
            <td>79.96</td>
        </tr>
        <tr>
            <td>Pre-finetuning</td>
            <td>88.59</td>
            <td>80.97</td>
            <td>84.58</td>
        </tr>
    </tbody>
</table>

Example command (ngpu=8):
```
CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/refcoco.json --batch_size 2 --lr_backbone 1e-5 --text_encoder_lr 5e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --ema --output-dir weights/$exp_id --load weights/pretrained_checkpoint.pth

CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/refcoco.json --batch_size 2 --lr_backbone 1e-5 --text_encoder_lr 5e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --ema --output-dir weights/$exp_id --load weights/prefinetune_refcoco_checkpoint.pth --eval --test --test_type testA
```

### Phrase grounding
The config file for pretraining is ``configs/flickr.json``. For model inference, use the input arguments ``--eval --test``.

Weights: [Separate](https://unitab.blob.core.windows.net/data/weights/separate_flickrGrounding_checkpoint.pth), [Pre-finetuning](https://unitab.blob.core.windows.net/data/weights/prefinetune_flickrGrounding_checkpoint.pth).

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Flickr</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Separate</td>
            <td>79.39</td>
        </tr>
        <tr>
            <td>Pre-finetuning</td>
            <td>79.58</td>
        </tr>
    </tbody>
</table>

Example command (ngpu=8):
```
CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/flickr.json --batch_size 2 --lr_backbone 1e-5 --text_encoder_lr 5e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --ema --do_flickrgrounding --output-dir weights/$exp_id --load weights/pretrained_checkpoint.pth

CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/flickr.json --batch_size 2 --lr_backbone 1e-5 --text_encoder_lr 5e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --ema --do_flickrgrounding --output-dir weights/$exp_id --load weights/prefinetune_flickrGrounding_checkpoint.pth --eval --test
```

### COCO captioning
The config file for pretraining is ``configs/flickr_cococaption.json``. For model inference, use the input arguments ``--eval --test``. 

For the final number, an output prediction json file will be automatically stored at ``weights/$exp_id/results/pred_dict_$CIDEr.json``. Please follow the official evaluation for [COCO captioning](https://github.com/tylin/coco-caption) evaluation. We will better intergrate the caption evaluations in future versions.

Weights: [Separate](https://unitab.blob.core.windows.net/data/weights/separate_MScococaption_checkpoint.pth), [Pre-finetuning](https://unitab.blob.core.windows.net/data/weights/prefinetune_MScococaption_checkpoint.pth).

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>CIDEr</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Separate</td>
            <td>119.3</td>
        </tr>
        <tr>
            <td>Pre-finetuning</td>
            <td>119.8</td>
        </tr>
    </tbody>
</table>

Example command (ngpu=16):
```
CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/flickr_cococaption.json --lr_backbone 2e-5 --text_encoder_lr 2e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --do_caption --no_detection --ema --output-dir weights/$exp_id --load weights/pretrained_checkpoint.pth

CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/flickr_cococaption.json --lr_backbone 2e-5 --text_encoder_lr 2e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --do_caption --no_detection --ema --output-dir weights/$exp_id --load weights/prefinetune_MScococaption_checkpoint.pth --eval --test
```

### Visual question answering on VQAv2
The config file for pretraining is ``configs/flickr_vqav2caption.json`` and ``configs/flickr_vqav2captionKP.json``. Adjust the ``GT_type`` between ``vqav2caption`` and ``vqav2captionKP`` for std and KP splits. For model inference, use the input arguments ``--eval --test``. 

For the final number, an output prediction json file will be automatically stored at ``weights/$exp_id/results/pred_dict_$CIDEr.json``. Please follow the official evaluation for [VQAv2](https://visualqa.org/evaluation.html) evaluation. We will better intergrate the caption evaluations in future versions.

Weights: [Separate](https://unitab.blob.core.windows.net/data/weights/separate_VQAv2_checkpoint.pth), [Pre-finetuning](https://unitab.blob.core.windows.net/data/weights/prefinetune_VQAv2_checkpoint.pth). KP split: [Separate](https://unitab.blob.core.windows.net/data/weights/separate_VQAv2KP_checkpoint.pth), [Pre-finetuning](https://unitab.blob.core.windows.net/data/weights/prefinetune_VQAv2KP_checkpoint.pth).


<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>test-dev</th>
            <th>KP-test</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Separate</td>
            <td>69.9</td>
            <td>66.6</td>
        </tr>
        <tr>
            <td>Pre-finetuning</td>
            <td>70.7</td>
            <td>67.5</td>
        </tr>
    </tbody>
</table>

Example command (ngpu=16):
```
CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/flickr_vqav2caption.json --lr_backbone 2e-5 --text_encoder_lr 2e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --do_caption --no_detection --ema --output-dir weights/$exp_id --load weights/pretrained_checkpoint.pth

CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/flickr_vqav2caption.json --lr_backbone 2e-5 --text_encoder_lr 2e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --do_caption --no_detection --ema --output-dir weights/$exp_id --load weights/prefinetune_VQAv2_checkpoint.pth --eval --test
```

## Acknowledgement
The project is built based on the following repository:
* [MDETR--Modulated Detection for End-to-End Multi-Modal Understanding](https://github.com/ashkamath/mdetr),
* [transformers](https://github.com/huggingface/transformers).
* [ColossalAI](https://github.com/hpcaitech/ColossalAI)
