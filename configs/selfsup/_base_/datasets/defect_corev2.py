# dataset settings
data_source = 'DefectData'
dataset_type = 'ThreeViewDataset'  ## change
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# The difference between mocov2 and mocov1 is the transforms in the pipeline

train_pipeline = [
    dict(type='RandomResizedCrop', size=192, scale=(0.2, 1.)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.5),
    dict(type='RandomHorizontalFlip'),
]

## copy simmim config
train_rec_pipeline = [
    dict(
        type='RandomResizedCrop',
        size=192,
        scale=(0.67, 1.0),
        ratio=(3. / 4., 4. / 3.)),
    dict(type='RandomHorizontalFlip')
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])
    train_rec_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

## refer to simmim config
train_rec_pipeline.append(
    dict(
        type='BlockwiseMaskGenerator',
        input_size=192,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.6))

# dataset summary
data = dict(
    samples_per_gpu=16,  # total 32*8=256
    workers_per_gpu=12,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='/home/dataE/pycharmproject/why/mmselfsup_0.7.0/data/Defect/train',
            ann_file='/home/dataE/pycharmproject/why/mmselfsup_0.7.0/data/Defect/meta/train.txt',
        ),
        num_views=[2,1],
        pipelines=[train_pipeline,train_rec_pipeline],
        prefetch=prefetch,
    ))
