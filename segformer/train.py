from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor

# Define the dataset
dataset = dict(
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='CustomDataset', # replace with your own dataset type
            data_root='data/train',
            img_dir='images',
            ann_dir='labels',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='Resize', img_scale=(480, 640), ratio_range=(0.5, 2.0)),
                dict(type='RandomCrop', crop_size=(480, 640), cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                dict(type='Pad', size=(480, 640), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ]
        )
    ),
)



# Build the SSFormer model
cfg = model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SSFormer',
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_encoder_layers=12,
        num_decoder_layers=12,
        input_resolution=(480, 640),
        drop_rate=0.0,
        transformer_activation='gelu'
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=768,
        channels=256,
        in_index=0,
        kernel_size=3,
        num_convs=2,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=2, 
        norm_cfg=dict(type='BN')
    ),
    # model training and testing settings
    train_cfg=dict(),
    #test_cfg=dict(mode='slide', crop_size=(480, 640), stride=(320, 320))
)
model = build_segmentor(cfg)


#https://github.com/shiwt03/SSformer/blob/main/configs/SSformer/SSformer_swin_768x768_80k_Cityscapes.py