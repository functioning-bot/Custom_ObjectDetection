from typing import Callable, Dict, List, Optional, Union
import torch
from torch import nn, Tensor
import torchvision
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool

from torchvision.models import resnet #, resnet50, ResNet50_Weights
from torchvision.models import efficientnet
from torchvision.models import get_model, get_model_weights, get_weight, list_models

def get_backbone(model_name: str,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ):
    weights_enum = get_model_weights(model_name)
    weights = weights_enum.DEFAULT #IMAGENET1K_V1
    #weights = ResNet50_Weights.DEFAULT
    if model_name.startswith('resnet'):
        backbone = resnet.__dict__[model_name](weights=weights, norm_layer=norm_layer)
    elif model_name.startswith('efficientnet'):
        backbone = efficientnet.__dict__[model_name](weights=weights, norm_layer=norm_layer)
    elif model_name.startswith('swin'):
        backbone = get_model(model_name)
    else:
        backbone = get_model(model_name)

    return backbone
    # weights_backbone = ResNet50_Weights.verify(weights)
    # backbone = resnet50(weights=weights_backbone, progress=True)


class MyBackboneWithFPN(nn.Module):
    def __init__(
        self,
        model_name: str, #= 'resnet50'
        trainable_layers: int,
        #return_layers: Dict[str, str],
        #in_channels_list: List[int],
        out_channels: int = 256, #the number of channels in the FPN
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        weights_enum = get_model_weights(model_name) #ResNet152_Weights
        weights = weights_enum.DEFAULT #ResNet152_Weights.IMAGENET1K_V2
        #weights = ResNet50_Weights.DEFAULT
        
        # Support both ResNet and other backbones
        if model_name.startswith('resnet'):
            backbone = resnet.__dict__[model_name](weights=weights, norm_layer=norm_layer)
        else:
            backbone = get_backbone(model_name, norm_layer)
        # weights_backbone = ResNet50_Weights.verify(weights)
        # backbone = resnet50(weights=weights_backbone, progress=True)

        #trainable_layers =2
        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers] #trainable_layers=0=>layers_to_train=[]
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
        
        returned_layers = [1, 2, 3, 4]
        #return_layers (Dict[name, new_name]): a dict containing the names of the modules for which the activations will be returned as the key of the dict
        return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)} #{'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        in_channels_stage2 = backbone.inplanes // 8 #2048//8=256
        #in_channels_list:List[int] number of channels for each feature map that is returned, in the order they are present in the OrderedDict
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
        #[256, 512, 1024, 2048]
        # BackboneWithFPN(
        #     backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
        # )
        #return_layers={'layer1': 'feat1', 'layer3': 'feat2'} #[name, new_name]
        #https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py
        self.body = torchvision.models._utils.IntermediateLayerGetter(backbone, return_layers=return_layers)
        # >>> out = new_m(torch.rand(1, 3, 224, 224))
        #     >>> print([(k, v.shape) for k, v in out.items()])
        #     >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        #     >>>      ('feat2', torch.Size([1, 256, 14, 14]))]

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.body(x) #[16, 3, 800, 1344]
        x = self.fpn(x)
        return x
    
    #not used
    def create_fpnbackbone(self, backbone, trainable_layers):
        #backbone = get_model(backbone_modulename, weights="DEFAULT")
        trainable_layers =2
        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        
        extra_blocks = LastLevelMaxPool()
        returned_layers = [1, 2, 3, 4]
        #return_layers (Dict[name, new_name]): a dict containing the names of the modules for which the activations will be returned as the key of the dict
        return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
        in_channels_stage2 = backbone.inplanes // 8
        #in_channels_list:List[int] number of channels for each feature map that is returned, in the order they are present in the OrderedDict
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
        #the number of channels in the FPN
        out_channels = 256
        # BackboneWithFPN(
        #     backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
        # )
        #return_layers={'layer1': 'feat1', 'layer3': 'feat2'} #[name, new_name]
        #https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py
        body = torchvision.models._utils.IntermediateLayerGetter(backbone, return_layers=return_layers)
        # >>> out = new_m(torch.rand(1, 3, 224, 224))
        #     >>> print([(k, v.shape) for k, v in out.items()])
        #     >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        #     >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
        
        fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=None,
        )
        return body, fpn

class EfficientNetBackboneWithFPN(nn.Module):
    """
    EfficientNet backbone with FPN or BiFPN
    EfficientNet has different layer structure than ResNet
    """
    def __init__(
        self,
        model_name: str,  # e.g., 'efficientnet_b3'
        trainable_layers: int,
        out_channels: int = 256,
        use_bifpn: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        
        # Load EfficientNet backbone
        weights_enum = get_model_weights(model_name)
        weights = weights_enum.DEFAULT
        backbone = efficientnet.__dict__[model_name](weights=weights, norm_layer=norm_layer)
        
        # EfficientNet structure: features[0-8] are stages
        # We'll extract features from stages 3, 4, 5, 6 (similar to C2, C3, C4, C5)
        # For EfficientNet-B3:
        #   Stage 3: 48 channels
        #   Stage 4: 96 channels
        #   Stage 5: 136 channels
        #   Stage 6: 232 channels
        #   Stage 7: 384 channels
        
        # Determine channel sizes based on model
        if 'b0' in model_name:
            in_channels_list = [40, 112, 320]
            feature_indices = [2, 4, 8]
        elif 'b1' in model_name:
            in_channels_list = [40, 112, 320]
            feature_indices = [2, 4, 8]
        elif 'b2' in model_name:
            in_channels_list = [48, 120, 352]
            feature_indices = [2, 4, 8]
        elif 'b3' in model_name:
            in_channels_list = [48, 136, 384]
            feature_indices = [2, 4, 8]
        elif 'b4' in model_name:
            in_channels_list = [56, 160, 448]
            feature_indices = [2, 4, 8]
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {model_name}")
        
        # Freeze layers
        if trainable_layers < 8:
            for i, (name, parameter) in enumerate(backbone.named_parameters()):
                if i < (len(list(backbone.parameters())) - trainable_layers * 10):
                    parameter.requires_grad_(False)
        
        # Create feature extractor
        self.body = backbone.features
        
        # Create FPN or BiFPN
        if use_bifpn:
            from DeepDataMiningLearning.detection.bifpn import BiFPN
            self.fpn = BiFPN(in_channels_list, out_channels, num_layers=3)
        else:
            extra_blocks = LastLevelMaxPool()
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=extra_blocks,
                norm_layer=norm_layer,
            )
        
        self.feature_indices = feature_indices
        self.out_channels = out_channels
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Extract features from specific stages
        features = []
        for idx, layer in enumerate(self.body):
            x = layer(x)
            if idx in self.feature_indices:
                features.append(x)
        
        # Create ordered dict for FPN
        feature_dict = {str(i): feat for i, feat in enumerate(features)}
        
        # Apply FPN/BiFPN
        x = self.fpn(feature_dict)
        return x


import os
try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.") #pip install -q torchinfo

def remove_classificationheader(model, num_removeblock):
    modulelist=model.children() #resnet50(pretrained=True).children()
    num_removeblock = 0-num_removeblock #-2
    newbackbone = nn.Sequential(*list(modulelist)[:num_removeblock])
    return newbackbone


if __name__ == "__main__":
    os.environ['TORCH_HOME'] = '/data/cmpe249-fa23/torchhome/'
    DATAPATH='/data/cmpe249-fa23/torchvisiondata/'

    #model_name = 'resnet50' #["layer4", "layer3", "layer2", "layer1", "conv1"]
    #model_name = 'resnet152' #["layer4", "layer3", "layer2", "layer1", "conv1"]
    #https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py
    model_name = 'swin_s' # 'avgpool','flatten','head'
    backbone = get_model(model_name, weights="DEFAULT")
    backbone=remove_classificationheader(backbone, 3)
    summary(model=backbone, 
        input_size=(1, 3, 64, 64), #(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    ) 

    trainable_layers = 2
    out_channels = 256
    model = MyBackboneWithFPN(model_name,trainable_layers, out_channels)
    x=torch.rand(1,3,64,64) #image.tensors #[2, 3, 800, 1312] list of tensors x= torch.rand(1,3,64,64)
    output = model(x) 
    print([(k, v.shape) for k, v in output.items()])
    #[('0', torch.Size([1, 256, 16, 16])), ('1', torch.Size([1, 256, 8, 8])), ('2', torch.Size([1, 256, 4, 4])), ('3', torch.Size([1, 256, 2, 2])), ('pool', torch.Size([1, 256, 1, 1]))]
