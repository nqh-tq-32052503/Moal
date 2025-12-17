import copy
import logging
import torch
from torch import nn
from backbone.linears import SimpleLinear, SplitCosineLinear, CosineLinear, AC_Linear, CosineLinear2
import timm

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    # SimpleCIL or SimpleCIL w/ Finetune
    if name == "pretrained_vit_b16_224_adapter_dino":
        from backbone import vision_transformer_adapter
        from easydict import EasyDict
        ffn_num = args["ffn_num"]
        tuning_config = EasyDict(
            # AdaptFormer.
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=ffn_num,
            d_model=768,
            # VPT related
            vpt_on=False,
            vpt_num=0,
        )
        model = vision_transformer_adapter.vit_base_patch16_224_adapter_dino(num_classes=0,
            global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_adapter_ibot":
        from backbone import vision_transformer_adapter
        from easydict import EasyDict
        ffn_num = args["ffn_num"]
        tuning_config = EasyDict(
            # AdaptFormer.
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=ffn_num,
            d_model=768,
            # VPT related
            vpt_on=False,
            vpt_num=0,
        )
        model = vision_transformer_adapter.vit_base_patch16_224_adapter_ibot(num_classes=0,
            global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
        model.out_dim = 768
        return model.eval()

    elif '_adapter' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "adapt_ac_com_sdc_ema_auto" :
            from backbone import vision_transformer_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer.
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
            )
            if name == "pretrained_vit_b16_224_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name == "pretrained_vit_b16_224_in21k_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name == "vit_base_patch16_224_bilora_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_bilora(pretrained=True)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        elif args["model_name"] == "bilora_ac_com_sdc_ema_auto" :
            from backbone import vision_transformer_adapter
            model = vision_transformer_adapter.vit_base_patch16_224_bilora(pretrained=True)
            model.out_dim=768
            return model.eval()
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        elif 'bilora' in args['backbone_type']:
            self.model_type = 'bilora'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x, task=None):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        elif self.model_type == 'bilora':
            x = self.backbone(x, task=task)
            out = self.fc(x)
            out.update({"features": x})
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class FOSTERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(FOSTERNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.backbones = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None
        self.args = args

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.backbones)

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim :])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        self.backbones.append(get_backbone(self.args, self.pretrained))
        if self.out_dim is None:
            self.out_dim = self.backbones[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.backbones[-1].load_state_dict(self.backbones[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_backbone(self):
        for param in self.backbones.parameters():
            param.requires_grad = False
        self.backbones.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        print("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["backbone_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.backbones) == 1
        self.backbones[0].load_state_dict(model_infos['backbone'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc

class AdaptiveNet(nn.Module):
    def __init__(self, args, pretrained):
        super(AdaptiveNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.TaskAgnosticExtractor , _ = get_backbone(args, pretrained) #Generalized blocks
        self.TaskAgnosticExtractor.train()
        self.AdaptiveExtractors = nn.ModuleList() #Specialized Blocks
        self.pretrained=pretrained
        self.out_dim=None
        self.fc = None
        self.aux_fc=None
        self.task_sizes = []
        self.args=args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim*len(self.AdaptiveExtractors)
    
    def extract_vector(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        out=self.fc(features) #{logits: self.fc(features)}

        aux_logits=self.aux_fc(features[:,-self.out_dim:])["logits"] 

        out.update({"aux_logits":aux_logits,"features":features})
        out.update({"base_features":base_feature_map})
        return out
                
        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''
        
    def update_fc(self,nb_classes):
        _ , _new_extractor = get_backbone(self.args, self.pretrained)
        if len(self.AdaptiveExtractors)==0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
            self.AdaptiveExtractors[-1].load_state_dict(self.AdaptiveExtractors[-2].state_dict())

        if self.out_dim is None:
            # print(self.AdaptiveExtractors[-1])
            self.out_dim=self.AdaptiveExtractors[-1].out_dim        
        fc = self.generate_fc(self.feature_dim, nb_classes)             
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc=self.generate_fc(self.out_dim,new_task_size+1)
 
    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["backbone_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        checkpoint_name = checkpoint_name.replace("memo_", "")
        model_infos = torch.load(checkpoint_name)
        model_dict = model_infos['backbone']
        assert len(self.AdaptiveExtractors) == 1

        base_state_dict = self.TaskAgnosticExtractor.state_dict()
        adap_state_dict = self.AdaptiveExtractors[0].state_dict()

        pretrained_base_dict = {
            k:v
            for k, v in model_dict.items()
            if k in base_state_dict
        }

        pretrained_adap_dict = {
            k:v
            for k, v in model_dict.items()
            if k in adap_state_dict
        }

        base_state_dict.update(pretrained_base_dict)
        adap_state_dict.update(pretrained_adap_dict)

        self.TaskAgnosticExtractor.load_state_dict(base_state_dict)
        self.AdaptiveExtractors[0].load_state_dict(adap_state_dict)
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc



class SimpleVitNet_AC(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x)
        out.update({"features": x})
        return out


class SimpleVitNet_AL(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.ac_model = None

    def update_fc(self, nb_classes, Hidden):
        ac_model = self.generate_fc(self.feature_dim, Hidden, nb_classes).to(self._device)

        if self.ac_model is not None:
            nb_output = self.ac_model.out_features
            hidden_weight = copy.deepcopy(self.ac_model.fc[0].weight.data)
            ac_model.fc[0].weight = nn.Parameter(hidden_weight.float())

            weight = copy.deepcopy(self.ac_model.fc[-1].weight.data)
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, Hidden).to(self._device)])

            ac_model.fc[-1].weight = nn.Parameter(weight.float())

        del self.ac_model
        self.ac_model = ac_model

    def generate_fc(self, in_dim, Hidden, out_dim):
        fc = AC_Linear(in_dim, Hidden, out_dim)
        return fc

    def extract_vector(self, x, task=None):
        return self.backbone(x, task=task)

    def forward(self, x, task=None):
        x = self.backbone(x, task=task)
        if self.ac_model == None:
            out = self.fc(x)
        else:
            out = self.ac_model(x)
        out.update({"features": x})
        # out.update({"train_logits": out["logits"]})
        return out

class BiLoRAIncNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.args = args

    def init_ac_fc(self):
        self.ac_model = None
        self.current_task = 0
        self.list_fc = nn.ModuleList([CosineLinear(self.feature_dim, self.args["init_cls"])])
        self.list_ac = nn.ModuleList()
    
    def update_task(self):
        self.current_task += 1

    def update_fc(self, cosine_fc = False):
        if cosine_fc:
            fc = CosineLinear2(self.feature_dim, self.args["increment"]).to(self._device)
            self.list_fc.append(fc)
        else:
            ac_model = AC_Linear(self.feature_dim, self.args["Hidden"], self.args["increment"]).to(self._device)
            self.list_ac.append(ac_model)
    
    def forward(self, x, task=None):
        x = self.backbone(x, task=task)
        if self.current_task == 1:
            out = self.list_fc[0](x)
            out.update({"features": x})
            return out
        else:
            fc_heads = self.list_fc[:self.current_task]
            ac_heads = self.list_ac[:self.current_task]
            fc_logits = [head(x) for head in fc_heads]
            fc_out = torch.cat([fc_logit['train_logits'] for fc_logit in fc_logits], dim=1)
            train_out = {'train_logits': fc_out}
            ac_logits = [head(x) for head in ac_heads]
            ac_out = torch.cat([ac_logit['logits'] for ac_logit in ac_logits], dim=1)
            out = {'logits': ac_out}
            out.update({"features": x})
            out.update({"train_logits": train_out["train_logits"]})
            return out