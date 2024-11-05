
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet152, ResNet152_Weights


import torch
from torch import nn







def get_network() -> nn.Module:
    #! now we only use the resnet152 model from torchvision
    model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
    
    return model


class BatchNormLinear(nn.Module):
    def __init__(self, in_features, out_features, device='cuda'):
        super(BatchNormLinear, self).__init__()
        self.batch_norm = nn.BatchNorm1d(in_features, device=device)
        self.linear = nn.Linear(in_features, out_features, device=device)

    def forward(self, x):
        x = self.batch_norm(x)
        return self.linear(x)
    
    

class TargetModel(nn.Module):
    def __init__(self, imported_model, multichannel_fn, classes=10, resolutions=[32, 16, 8, 4]):
        super(TargetModel, self).__init__()
        self.imported_model = imported_model
        self.multichannel_fn = multichannel_fn
        self.classes = classes
        # self.fix_seed = False


        # Define all layer dimensions
        self.all_dims = [
            3 * 224 * 224 * len(resolutions),
            64 * 56 * 56,
            *[256 * 56 * 56] * len(imported_model.layer1),
            *[512 * 28 * 28] * len(imported_model.layer2),
            *[1024 * 14 * 14] * len(imported_model.layer3),
            *[2048 * 7 * 7] * len(imported_model.layer4),
            2048,
            1000,
        ]

        # Create linear layers for each dimension
        self.linear_layers = torch.nn.ModuleList([
            BatchNormLinear(self.all_dims[i], classes, device="cuda") for i in range(len(self.all_dims))
        ])
        # self.linear_layers = nn.ModuleList([
        #     self._create_batchnorm_linear(self.all_dims[i], classes) for i in range(len(self.all_dims))
        # ])

    def _layer_operations(self, imported_model):
        # Define layer operations from imported model
        self.layer_operations = [
            nn.Sequential(
                imported_model.conv1,
                imported_model.bn1,
                imported_model.relu,
                imported_model.maxpool,
            ),
            *imported_model.layer1,
            *imported_model.layer2,
            *imported_model.layer3,
            *imported_model.layer4,
            imported_model.avgpool,
            imported_model.fc,
        ]

        
    def _create_batchnorm_linear(self, in_features, out_features):
        """Helper function to create a BatchNorm + Linear block."""
        return nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, out_features)
        )

    def prepare_input(self, x):
        """Prepare input by applying multichannel function, resizing, and normalizing."""
        x = self.multichannel_fn(x)
        # import pdb; pdb.set_trace()
        x = F.interpolate(x, size=(224, 224), mode='bicubic')
        x = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406] * (x.shape[1] // 3),
            std=[0.229, 0.224, 0.225] * (x.shape[1] // 3)
        )(x)
        return x

    def forward_until(self, x, layer_id):
        """Forward pass up to a specified layer."""
        x = self.prepare_input(x)
        for l in range(layer_id):
            
            if list(x.shape)[1:] == [2048, 1, 1]:
                x = x.reshape([-1, 2048])
            x = self.layer_operations[l](x)

        return x


    def predict_from_layer(self, x, l):
        """Predict from a specific layer."""
        x = self.forward_until(x, l)
        x = x.reshape([x.shape[0], -1])
        # import pdb; pdb.set_trace()
        return self.linear_layers[l](x)

    def predict_from_several_layers(self, x, layers):
        """Predict from several layers."""
        # x = x.double()        
        x = self.prepare_input(x)
        outputs = dict()
        # self.linear_layers[0] = self.linear_layers[0].double()
        outputs[0] = self.linear_layers[0](x.reshape([x.shape[0], -1]))
        # self.linear_layers[0] = self.linear_layers[0].float()
        for l in range(len(self.layer_operations)):
            if list(x.shape)[1:] == [2048, 1, 1]:
                x = x.reshape([-1, 2048])
            # self.layer_operations[l] = self.layer_operations[l].double()
            x = self.layer_operations[l](x)
            # self.layer_operations[l] = self.layer_operations[l].float()
            if l in layers:
                # self.linear_layers[l + 1] = self.linear_layers[l + 1].double()
                outputs[l + 1] = self.linear_layers[l + 1](x.reshape([x.shape[0], -1]))
                # self.linear_layers[l + 1] = self.linear_layers[l + 1].float()
                

        
        return outputs
    
    def forward(self, x):
        """Main forward pass, combining multiple layer predictions."""
        all_logits = self.predict_from_several_layers(x, [l - 1 for l in [0, 1, 5, 10, 20, 30, 35, 40, 45, 50, 52][1:]])
        # Add prediction from the backbone model itself
        x_ = self.prepare_input(x)
        # import pdb; pdb.set_trace()
        # self.imported_model.double()  
        # x_ = x_.double()                
        all_logits[54] = self.imported_model(x_)
        # self.imported_model.float()
        #! now all_logits matchs, but stack_logits is not
        stack_logits = torch.stack([all_logits[l] for l in [20, 30, 35, 40, 45, 50, 52, 54]], dim=1)
        # print("sum:", [all_logits[l][0].sum().item() for l in [20, 30, 35, 40, 45, 50, 52, 54]])
        # import pdb; pdb.set_trace()
        # Normalize and extract logits
        stack_logits = stack_logits - torch.max(stack_logits, dim=2, keepdim=True).values
        stack_logits = stack_logits - torch.max(stack_logits, dim=1, keepdim=True).values
        
        logits = torch.topk(stack_logits, 3, dim=1).values[:, 2]
        # logits = logits.float()
        return logits   # [bs, 100]
    
    def forward_original(self, x):
        x = self.multichannel_fn(x)
        x = F.interpolate(x, size=(224, 224), mode='bicubic')
        x = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406] * (x.shape[1] // 3),
            std=[0.229, 0.224, 0.225] * (x.shape[1] // 3)
        )(x)
        # import pdb; pdb.set_trace()
        x = self.imported_model(x)
        return x


class SourceModel(TargetModel):
    def __init__(self, imported_model, multichannel_fn, classes=10, resolutions=[32, 16, 8, 4]):
        super(SourceModel, self).__init__(imported_model, multichannel_fn, classes, resolutions)
        # self.fix_seed = True
        
    def get_logits_from_layer(self, x, l):
        """Predict from a specific layer."""
        x = self.forward_until(x, l)
        x = x.reshape([x.shape[0], -1])
        return self.linear_layers[l](x)
    
    def get_logits_from_several_layers(self, x, layers):

        all_logits = self.predict_from_several_layers(x, [l - 1 for l in [0, 1, 5, 10, 20, 30, 35, 40, 45, 50, 52][1:]])
        # Add prediction from the backbone model itself
        all_logits[54] = self.imported_model(self.prepare_input(x))

        # Stack logits from specific layers, [20,30,35,40,45,50,52]
        all_logits = torch.stack([all_logits[l] for l in [20, 30, 35, 40, 45, 50, 52, 54]], dim=1)
        

        return torch.mean(all_logits, dim=1)
    
    def forward(self, x):
        return self.get_logits_from_several_layers(x, [20, 30, 35, 40, 45, 50, 52])

    # def forward(self, x):
    #     #! debug if we turn ensemble off
    #     import pdb; pdb.set_trace()
    #     debug = False
    #     if debug:
    #         return self.forward_original(x)
    #     else: 
    #         """Main forward pass, combining multiple layer predictions."""
    #         all_logits = self.predict_from_several_layers(x, [l - 1 for l in [0, 1, 5, 10, 20, 30, 35, 40, 45, 50, 52][1:]])
    #         # Add prediction from the backbone model itself
    #         all_logits[54] = self.imported_model(self.prepare_input(x))

    #         # Stack logits from specific layers, [20,30,35,40,45,50,52]
    #         all_logits = torch.stack([all_logits[l] for l in [20, 30, 35, 40, 45, 50, 52, 54]], dim=1)
    #         cross_max = True
    #         if cross_max:
    #             # Normalize and extract logits
    #             all_logits = all_logits - torch.max(all_logits, dim=2, keepdim=True).values
    #             all_logits = all_logits - torch.max(all_logits, dim=1, keepdim=True).values
    #             # logits = torch.mean(all_logits, dim=1)
    #             logits = torch.topk(all_logits, 3, dim=1).values[:, 2]
    #         else:
    #             # average the logits
    #             logits = torch.mean(all_logits, dim=1)

    #         return logits   # [bs, 100]