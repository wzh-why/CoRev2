# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.utils import build_from_cfg
from torchvision.transforms import Compose

from .base import BaseDataset
from .builder import DATASETS, PIPELINES, build_datasource
from .utils import to_numpy
from .pipelines import montage_PIL,random_choice,random_subfigure_cutout

@DATASETS.register_module()
class RandomchoiceAndRandcutView(BaseDataset):  ##
    """The dataset outputs multiple views of an image.

    The number of views in the output dict depends on `num_views`. The
    image can be processed by one pipeline or multiple piepelines.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        num_views (list): The number of different views.
        pipelines (list[list[dict]]): A list of pipelines, where each pipeline
            contains elements that represents an operation defined in
            `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.

    Examples:
        >>> dataset = MultiViewDataset(data_source, [2], [pipeline])
        >>> output = dataset[idx]
        The output got 2 views processed by one pipeline.

        >>> dataset = MultiViewDataset(
        >>>     data_source, [2, 6], [pipeline1, pipeline2])
        >>> output = dataset[idx]
        The output got 8 views processed by two pipelines, the first two views
        were processed by pipeline1 and the remaining views by pipeline2.
    """

    def __init__(self, data_source, num_views, pipelines, prefetch=False):
        assert len(num_views) == len(pipelines)
        self.data_source = build_datasource(data_source)
        self.pipelines = []
        for pipe in pipelines:
            pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            self.pipelines.append(pipeline)
        self.prefetch = prefetch

        trans = []
        assert isinstance(num_views, list)
        for i in range(len(num_views)):
            trans.extend([self.pipelines[i]] * num_views[i])

        ## src trans all to tensor and normlize
        self.trans = trans

    def __getitem__(self, idx):

        img = self.data_source.get_img(idx)

        img_2 = montage_PIL(img)

        img_3,order = random_choice(img)

        img_4,mask = random_subfigure_cutout(img)

        ##
        multi_views = list(map(lambda trans: trans(img), self.trans))
        multi_views.extend([img_2,img_3,img_4])

        # print('multi len is {}'.format(len(multi_views)))

        # multi_views = list(map(lambda trans: trans(img), self.trans))
        # if self.prefetch:
        #     multi_views = [
        #         torch.from_numpy(to_numpy(img)) for img in multi_views
        #     ]
        return dict(img=multi_views,order=order,mask=mask)

    def evaluate(self, results, logger=None):
        return NotImplemented