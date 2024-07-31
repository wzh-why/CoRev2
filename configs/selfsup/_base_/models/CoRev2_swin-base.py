# model settings
model = dict(
    type='CoRev2',
    backbone=dict(
        type='CoRev2SwinTransformer',
        arch='B',
        img_size=224, ###
        stage_cfgs=dict(block_cfgs=dict(window_size=6))), ## 不需要修改

    neck=dict(type='SimMIMNeck', in_channels=128 * 2**3, encoder_stride=32),   ### in_channel 128还是192 取决于 arch='B'还是 ‘L’
    head=dict(type='SimMIMHead', patch_size=4, encoder_in_channels=3),

    ### 添加对比neck 以及 对比head
    base_momentum=0.99,
    contra_weight=1.0,
    mim_weight= 1.0,
    contra_neck = dict(  ##投射头 refer to mocov3
        type='CoRev2ContraNeck',
        in_channels=1024,  ###修改
        hid_channels=2048, ###修改为2048
        out_channels=256,
        num_layers=2,
        with_bias=False,
        with_last_bn=True,
        with_last_bn_affine=False,
        with_last_bias=False,
        with_avg_pool=False, # 全局池化在backbone阶段进行
        vit_backbone=False),
    contra_head = dict( ## 
        type='CoRev2ContraHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            num_layers=2,
            with_bias=False,
            with_last_bn=True,
            with_last_bn_affine=False,
            with_last_bias=False,
            with_avg_pool=False),
        temperature=0.2))
