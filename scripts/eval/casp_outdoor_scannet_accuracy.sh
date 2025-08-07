ckpt_path="weights/casp_outdoor.pth"
train_size=832
threshold=0.3
border_removal=0

python -m src.eval \
    experiment=casp_scannet \
    ckpt_path=${ckpt_path} \
    model.test_task=accuracy \
    model.net.config.train_size=${train_size} \
    model.net.config.threshold=${threshold} \
    model.net.config.border_removal=${border_removal}
