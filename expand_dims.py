import numpy as np
import torch
from torch_interp import interp
import copy


class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

def create_hooks(*layers):
    hooks = []
    for layer in layers:
        fw_hooks = [Hook(block) for block in layer.children()]
        bw_hooks = [Hook(block,backward=True) for block in layer.children()]
        hooks.append([fw_hooks,bw_hooks])
    return hooks

def create_feed(hooks,out_dims,nsplits,data,target,cuda=False):
    out_arr = []
    grad_arr = []
    out_arr.append(data.unsqueeze(0))
    for i in range(len(hooks)):
        fw_ = torch.stack([x.output.data for x in hooks[i][0]])
        bw_ = torch.stack([x.input[0].data for x in hooks[i][1]])
        size_ = list(fw_.size())
        x_old = np.linspace(1,out_dims[i],size_[0])
        if cuda:
            x_new = torch.from_numpy(np.linspace(1,out_dims[i],out_dims[i])).cuda().float()
        else:
            x_new = torch.from_numpy(np.linspace(1,out_dims[i],out_dims[i])).float()
        temp_size_fw = list(fw_.size())
        temp_size_bw = list(bw_.size())
        fw_ = fw_.view(fw_.size(0),-1)
        bw_ = bw_.view(bw_.size(0),-1)
        temp_size_fw[0] = out_dims[i]
        temp_size_bw[0] = out_dims[i]
        temp_size_fw = tuple(temp_size_fw)
        temp_size_bw = tuple(temp_size_bw)
        out_ = interp(fw_,x_old,x_new).view(temp_size_fw).float()
        grad_= interp(bw_,x_old,x_new).view(temp_size_bw).float()
        if nsplits[i] != 0:
            out_ = torch.chunk(out_,nsplits[i],dim=0)
            grad_ = torch.chunk(grad_,nsplits[i],dim=0)
            for j in range(len(grad_)):
                out_arr.append(out_[j][-1].unsqueeze(0))
                grad_arr.append(grad_[j][-1].unsqueeze(0))
        else:
            out_arr.append(out_[-1].unsqueeze(0))
            grad_arr.append(grad_[-1].unsqueeze(0))
    grad_arr.append(target.unsqueeze(0))
    return(out_arr,grad_arr)

def create_state(model,layers,out_dims,nsplits,cuda=False):
    state=copy.deepcopy(model.state_dict())
    type_dict = {}
    new_state = {}
    for param in model.state_dict():
        for layer in layers:
            if layer in param and ("weight" in param or "bias" in param):
                num_,rest = param.split('.',1)[1].split('.',1)
                key = layer + "," + rest  #key = layer1,conv1.weight (comma allows easy split later)
                if key not in type_dict:
                    type_dict[key]=[state[param].data] #will be in order, no need to check num_
                else:
                    type_dict[key].append(state[param].data)
        if param.split('.',1)[0] not in layers and ("weight" in param or "bias" in param):
            new_state[param] = state[param].data
    for key in list(type_dict):
        for i in range(len(layers)):
            if layers[i] in key:
                if len(type_dict[key])!=1:
                    x_old = np.linspace(1,out_dims[i]-1,len(type_dict[key])-1)
                    if cuda:
                        x_new = torch.from_numpy(np.linspace(1,out_dims[i]-1,out_dims[i]-1)).cuda().float()
                    else:
                        x_new = torch.from_numpy(np.linspace(1,out_dims[i]-1,out_dims[i]-1)).float()
                    first_value = type_dict[key][0]
                    type_dict[key].pop(0)
                    vl = torch.stack(type_dict[key])
                    temp_size = list(vl.size())
                    vl = vl.view(vl.size(0),-1)
                    temp_size[0] = out_dims[i]-1
                    temp_size = tuple(temp_size)
                    weight_ = interp(vl, x_old, x_new).view(temp_size).float()
                    if nsplits[i]==0:
                        weight_ = list(weight_)
                        weight_.insert(0,first_value)
                        type_dict[key] = weight_
                    else:
                        weight_ = list(weight_)
                        weight_.insert(0,torch.zeros_like(weight_[0])) #dummy tensor for chunk
                        weight_ = torch.chunk(torch.stack(weight_),nsplits[i],dim=0)
                        weight_0 = list(weight_[0])
                        weight_0[0] = first_value  #overwrite dummy tensor
                        type_dict[key]=weight_0 
                        for j in range(1,len(weight_)): #should be 1
                            new_key = "repeat"+key
                            type_dict[new_key]=list(weight_[j])
    for key in type_dict:
        lay_,rest = key.split(',',1)
        for w in range(len(type_dict[key])):
            new_key = lay_ + "." + str(w) + "." + rest
            new_state[new_key] = type_dict[key][w]
    for key in list(new_state):
        new_key = key.replace(".","_")
        new_state[new_key] = new_state.pop(key)
    return new_state