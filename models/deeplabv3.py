import torch
import torch.nn as nn
import torchvision.models as models


class DeeplabV3(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet50(
            pretrained=False,
            num_classes=n_class
        )
    
    def forward(self, x):
        return self.model(x)["out"]

def get_model(model_config):
    r"""Get DeeplabV3 model.
    
    Args:
        model_config (EasyDict): Contain model configuration.
    """
    return DeeplabV3(**model_config)


if __name__ == "__main__":
    from easydict import EasyDict
    
    model_config = EasyDict(dict(n_class=3))
    model = get_model(model_config)