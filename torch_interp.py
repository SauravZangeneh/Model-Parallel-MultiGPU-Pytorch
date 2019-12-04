import torch
import numpy as np

def interp(x,x_old,x_new,axis=0,method=1):
    x_size = x.size()
    x = x.view(x_size[axis],-1)
    y_out = []
    assert (x_old.size == x_size[axis]), "size should match along interpolation axis"
    if method == 1:
        for element in x_new.data:
            for i in range(x_old.size-1):
                if element<x_old[i+1] and element>x_old[i]:
                    slope = (x.data[i+1] - x.data[i])/(x_old[i+1] - x_old[i])
                    y_ = (element-x_old[i])*slope + x.data[i]
                    y_out.append(y_)
                    break
                elif element == x_old[i]:
                    y_out.append(x.data[i])
                    break
                elif element == x_old[i+1]:
                    y_out.append(x.data[i+1])
                    break
        y = torch.stack(y_out).float()
    else:
        slope =  (x.data[-1] - x.data[0])/(x_old[-1] - x_old[0])
        y = (x_new-x_old[0]).unsqueeze(1)*slope.unsqueeze(0).float() + x.data[0]
    return y