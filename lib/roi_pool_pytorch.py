import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class RoIPool(nn.Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        """
        features shape is (batch_size, num_channels, img_height, img_width)
        rois shape is (num_rois, 5), where the first index is batch index, the last 4 indexes are the coordinate
        of the upper left corner and the lower right corner
        spatial scale should be like 1/16, 1/8, etc.
        """

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        outputs = Variable(torch.zeros(num_rois, num_channels, self.pooled_height, self.pooled_width)).cuda()

        for roi_idx, roi in enumerate(rois):
            batch_idx = int(roi[0])
            if batch_idx > batch_size - 1:
                raise ValueError("Batch index out of range!")
            upleft_x, upleft_y, downright_x, downright_y = np.round(roi[1:].cpu().numpy() * self.spatial_scale).astype(int)
            roi_width = max(downright_x - upleft_x + 1, 1)
            roi_height = max(downright_y - upleft_y + 1, 1)
            bin_size_w = float(roi_width) / float(self.pooled_width)
            bin_size_h = float(roi_height) / float(self.pooled_height)

            for ph in range(self.pooled_height):
                hstart = int(np.floor(ph * bin_size_h))
                hend = int(np.ceil((ph + 1) * bin_size_h))
                hstart = min(data_height, max(0, hstart + upleft_y))
                hend = min(data_height, max(0, hend + upleft_y))

                for pw in range(self.pooled_width):
                    wstart = int(np.floor(pw * bin_size_w))
                    wend = int(np.ceil((pw + 1) * bin_size_w))
                    wstart = min(data_width, max(0, wstart + upleft_x))
                    wend = min(data_width, max(0, wend + upleft_x))
                    is_error = (hend <= hstart) or (wend <= wstart)

                    if is_error:
                        outputs[roi_idx, :, ph, pw] = 0

                    else:
                        data = features[batch_idx]
                        outputs[roi_idx, :, ph, pw] = torch.max(torch.max(data[:, hstart:hend, wstart:wend], dim=1)[0], dim=2)[0].view(-1)

        return outputs

