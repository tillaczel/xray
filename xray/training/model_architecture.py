from torchsummary import summary
import torch

from xray.training.models import Convnet


def get_summary():
    if torch.cuda.is_available():
        summary(Convnet().cuda(), (1, 64, 64), 256)