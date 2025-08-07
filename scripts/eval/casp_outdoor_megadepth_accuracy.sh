ckpt_path="weights/casp_outdoor.pth"
image_size=1152

python -m src.eval \
    experiment=casp_megadepth \
    ckpt_path=${ckpt_path} \
    model.test_task=accuracy \
    data.test_config.dataset_builder.image_size=${image_size}
