# Official PyTorch implementation of [DiG](https://arxiv.org/pdf/2207.00193)

This repository is built upon [MoCo V3](https://github.com/facebookresearch/moco-v3) and [MAE](https://github.com/pengzhiliang/MAE-pytorch), thanks very much!

## Data preparation
All datasets for pre-training and fine-tuning are processed from public datasets.

<table>
<tbody>
  <tr>
    <td>Unlabeled Real Data</td>
    <td><a href="https://1drv.ms/u/s!AgwG2MwdV23ckOhflPF53mo6a3lpsQ?e=6hoJv5" target="_blank" rel="noopener noreferrer">CC-OCR</a></td>
  </tr>
  <tr>
    <td>Synthetic Text Data</td>
    <td><a href="https://pan.baidu.com/s/1BMYb93u4gW_3GJdjBWSCSw&shfl=sharepset" target="_blank" rel="noopener noreferrer">SynthText, Synth90k</a> (Baiduyun with passwd: wi05)</td>
  </tr>
  <tr>
    <td>Annotated Real Data</td>
    <td><a href="https://1drv.ms/u/s!AgwG2MwdV23ckOdQdd4YekGsOUXGbw?e=qlbQRT" target="_blank" rel="noopener noreferrer">TextOCR</a>, <a href="https://1drv.ms/u/s!AgwG2MwdV23ckOdRteE0zcxINnlfJA?e=1iW92G" target="_blank" rel="noopener noreferrer">OpenImageTextV5</a></td>
  </tr>
  <tr>
    <td>Scene Text Recognition Benchmarks</td>
    <td><a href="https://1drv.ms/u/s!AgwG2MwdV23ckOhvBLdfDlYLNJaiIw?e=vh9krZ" target="_blank" rel="noopener noreferrer">IIIT5k</a>, <a href="https://1drv.ms/u/s!AgwG2MwdV23ckOhyQn60SzFI97IAeQ?e=Pk8rlZ" target="_blank" rel="noopener noreferrer">SVT</a>, <a href="https://1drv.ms/u/s!AgwG2MwdV23ckOhsX6uSXU9yLqjeoA?e=bes8bp" target="_blank" rel="noopener noreferrer">IC13</a>, <a href="https://1drv.ms/u/s!AgwG2MwdV23ckOhuy6ebkDhU3i5vcQ?e=t1XQN6" target="_blank" rel="noopener noreferrer">IC15</a>, <a href="https://1drv.ms/u/s!AgwG2MwdV23ckOhzwW9jeK0zajRwiA?e=ibLDvC" target="_blank" rel="noopener noreferrer">SVTP</a>, <a href="https://1drv.ms/u/s!AgwG2MwdV23ckOhq0MJ4-jHDq9gFaw?e=uaxaEX" target="_blank" rel="noopener noreferrer">CUTE</a>, <a href="https://1drv.ms/u/s!AgwG2MwdV23ckOhoiwC5wf4eC9kYoQ?e=oXzZNF" target="_blank" rel="noopener noreferrer">COCOText</a>, <a href="https://1drv.ms/u/s!AgwG2MwdV23ckOhp6ddoyLetHu2yaA?e=qTdZEc" target="_blank" rel="noopener noreferrer">CTW</a>, <a href="https://1drv.ms/u/s!AgwG2MwdV23ckOh02A7vn9kfCmuYjg?e=kkxmf6" target="_blank" rel="noopener noreferrer">Total-Text</a>, <a href="https://1drv.ms/u/s!AgwG2MwdV23ckOhw2Aj0lquBf3eGzA?e=pcFEth" target="_blank" rel="noopener noreferrer">HOST</a>, <a href="https://1drv.ms/u/s!AgwG2MwdV23ckOhxVi_7kppEkFMz2A?e=lKYfUY" target="_blank" rel="noopener noreferrer">WOST</a></td>
  </tr>
  <tr>
    <td>Handwritten Text Training Data</td>
    <td><a href="https://1drv.ms/u/s!AgwG2MwdV23ckOk19H2ZZLnzyGAf2g?e=w2WhRW" target="_blank" rel="noopener noreferrer">CVL</a>, <a href="https://1drv.ms/u/s!AgwG2MwdV23ckOh2pA6j5AL0Z0sVxQ?e=uNittd" target="_blank" rel="noopener noreferrer">IAM</a></td>
  </tr>
  <tr>
    <td>Handwritten Text Recognition Benchmarks</td>
    <td><a href="https://1drv.ms/u/s!AgwG2MwdV23ckOh4TIU1rmbcMSI2kg?e=jayq60" target="_blank" rel="noopener noreferrer">CVL</a>, <a href="https://1drv.ms/u/s!AgwG2MwdV23ckOh7VJ8vmfd7S_asCw?e=kdELCq" target="_blank" rel="noopener noreferrer">IAM</a></td>
  </tr>
</tbody>
</table>

## Setup

```
conda env create -f environment.yml
```

## Run
1. Pre-training
```bash
# Set the path to save checkpoints
OUTPUT_DIR='output/pretrain_dig'
# path to imagenet-1k train set
DATA_PATH='/path/to/pretrain_data/'


# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_mae_pretraining_moco.py \
        --image_alone_path ${DATA_PATH} \
        --mask_ratio 0.7 \
        --batch_size 128 \
        --opt adamw \
        --output_dir ${OUTPUT_DIR} \
        --epochs 10 \
        --warmup_steps 5000 \
        --max_len 25 \
        --num_view 2 \
        --moco_dim 256 \
        --moco_mlp_dim 4096 \
        --moco_m 0.99 \
        --moco_m_cos \
        --moco_t 0.2 \
        --num_windows 4 \
        --contrast_warmup_steps 0 \
        --contrast_start_epoch 0 \
        --loss_weight_pixel 1. \
        --loss_weight_contrast 0.1 \
        --only_mim_on_ori_img \
        --weight_decay 0.1 \
        --opt_betas 0.9 0.999 \
        --model pretrain_simmim_moco_ori_vit_small_patch4_32x128 \
        --patchnet_name no_patchtrans \
        --encoder_type vit \
```

2. Fine-tuning
```bash
# Set the path to save checkpoints
OUTPUT_DIR='output/'
# path to imagenet-1k set
DATA_PATH='/path/to/finetune_data'
# path to pretrain model
MODEL_PATH='/path/to/pretrain/checkpoint.pth'

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --master_port 10041 run_class_finetuning.py \
    --model simmim_vit_small_patch4_32x128 \
    --data_path ${DATA_PATH} \
    --eval_data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 256 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --data_set image_lmdb \
    --nb_classes 97 \
    --smoothing 0. \
    --max_len 25 \
    --epochs 10 \
    --warmup_epochs 1 \
    --drop 0.1 \
    --attn_drop_rate 0.1 \
    --drop_path 0.1 \
    --dist_eval \
    --lr 1e-4 \
    --num_samples 1 \
    --fixed_encoder_layers 0 \
    --decoder_name tf_decoder \
    --use_abi_aug \
    --num_view 2 \
```

3. Evaluation
```bash
# Set the path to save checkpoints
OUTPUT_DIR='output/'
# path to imagenet-1k set
DATA_PATH='/path/to/test_data'
# path to finetune model
MODEL_PATH='/path/to/finetune/checkpoint.pth'

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=$opt_nproc_per_node --master_port 10040 run_class_finetuning.py \
    --model simmim_vit_small_patch4_32x128 \
    --data_path ${DATA_PATH} \
    --eval_data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 512 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --data_set image_lmdb \
    --nb_classes 97 \
    --smoothing 0. \
    --max_len 25 \
    --resume ${MODEL_PATH} \
    --eval \
    --epochs 20 \
    --warmup_epochs 2 \
    --drop 0.1 \
    --attn_drop_rate 0.1 \
    --dist_eval \
    --num_samples 1000000 \
    --fixed_encoder_layers 0 \
    --decoder_name tf_decoder \
    --beam_width 0 \
```

## Result

|   model  | pretrain | finetune | average accuracy | weight |
|:--------:|:--------:|:--------:|:--------:| :--------:|
| vit-small |   10e   |   10e   |   85.21%  | [pretrain](https://1drv.ms/u/s!AgwG2MwdV23ckOhlLmStGZ03RSQLMA?e=WN9fJ9) [finetune](https://1drv.ms/u/s!AgwG2MwdV23ckOhm29tonOUPja4yXQ?e=ed3Cfs)|


## Citation
If you find this project helpful for your research, please cite the following paper:

```
@inproceedings{DiG,
  author    = {Mingkun Yang and
               Minghui Liao and
               Pu Lu and
               Jing Wang and
               Shenggao Zhu and
               Hualin Luo and
               Qi Tian and
               Xiang Bai},
  editor    = {Jo{\~{a}}o Magalh{\~{a}}es and
               Alberto Del Bimbo and
               Shin'ichi Satoh and
               Nicu Sebe and
               Xavier Alameda{-}Pineda and
               Qin Jin and
               Vincent Oria and
               Laura Toni},
  title     = {Reading and Writing: Discriminative and Generative Modeling for Self-Supervised
               Text Recognition},
  booktitle = {{MM} '22: The 30th {ACM} International Conference on Multimedia, Lisboa,
               Portugal, October 10 - 14, 2022},
  pages     = {4214--4223},
  publisher = {{ACM}},
  year      = {2022},
  url       = {https://doi.org/10.1145/3503161.3547784},
  doi       = {10.1145/3503161.3547784},
  timestamp = {Fri, 14 Oct 2022 14:25:06 +0200},
  biburl    = {https://dblp.org/rec/conf/mm/YangLLWZLTB22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## License
This project is under the CC-BY-NC 4.0 license. See [LICENSE](https://github.com/ayumiymk/DiG/blob/main/LICENSE) for details. If you are going to use it in a product, we suggest you [contact us](xbai@hust.edu.cn) regarding possible patent issues.
