### Training

create conda env

```bash
conda create -n gim python=3.8
conda activate gim
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pip==24.0
pip install -r requirements.txt
```

start training

```bash
# official
python train.py --num_nodes 1 --gpus 1 --max_epochs 10 --maxlen 938240 938240 938240 --lr 0.001 --min_lr 0.00005 --resample --img_size 832 --batch_size 1 --valid_batch_size 2

# MegaDepth + ScanNet
python train.py --num_nodes 1 --gpus 1 --max_epochs 10 --maxlen 938240 938240 --lr 0.001 --min_lr 0.00005 --resample --img_size 832 --batch_size 1 --valid_batch_size 2 --trains MegaDepth ScanNet
```
