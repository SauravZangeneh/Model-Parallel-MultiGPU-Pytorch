from __future__ import print_function
import argparse
import torch
import copy
from torchvision import datasets, transforms
from torch.autograd import Variable
import dni
import torch.multiprocessing as mp
import torch.distributed as dist
import platform
import time
from torch.utils.data import Dataset
from torch_modified_resnet import Dist_ResNet_MNIST, Bottleneck
import numpy as np
from torch_interp import interp
from expand_dims import create_hooks,create_feed,create_state

def initialize():
    ''' The goal is to train a small ResNet and interpolate the inputs to 
    each block as well as the weights
    '''
    dist_dataset=[]
    out_array = []
    grad_array = []
    batch_size=100
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../Mnist', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ])),
        batch_size=batch_size, shuffle=False)
#     train_loader = torch.utils.data.DataLoader(
#         datasets.CIFAR10('../CIFAR10', train=True, download=True,
#                transform=transforms.Compose([
#                    transforms.ToTensor(),
#                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#                ])),
#         batch_size=batch_size, shuffle=False)
    model = Dist_ResNet_MNIST(Bottleneck, [3, 3, 3, 3],dist_=False).cuda() #small resnet
    opt = torch.optim.Adam(model.parameters(),1e-3,weight_decay=1e-3)
    hooks = create_hooks(model.layer2,model.layer3) #creating hooks
    for i in range(1): #3
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            out=model(data.cuda())
            criterion = torch.nn.CrossEntropyLoss().cuda()
            loss=criterion(out,target.cuda())
            loss.backward()
            opt.step()
            opt.zero_grad()
            if i == 0:  #2
                out_arr,grad_arr=create_feed(hooks,[150,300],[0,2],data,target)
                if len(out_array)==0:
                    out_array, grad_array = out_arr,grad_arr
                else:
                    for j in range(len(out_array)):
                        out_array[j] = torch.cat((out_array[j],out_arr[j]),0)
                        grad_array[j] = torch.cat((grad_array[j],grad_arr[j]),0)
            if batch_idx == 0: 
                print ("Epoch: ",i," Loss : ",loss.data)
            break
    new_state = None #.  weight interpolation / and distribution across devices is slow 


#     new_state = create_state(model,["layer2","layer3"],[200,400],[0,2])

    return(out_array,grad_array,new_state)

def distribute_train(rank,size,block_type,cuda_):
    cust_val = torch.tensor(0)


    # We have to first get the shapes of the tensors (since it might vary) and send these to various devices
    x_dims = torch.zeros([5])
    g_dims = torch.zeros([5])
    if rank == size-1:
        g_dims = torch.zeros([2])
    if rank == 0:
        x_list,g_list,weights = initialize()
        for i in range(1,size):
            x_dims = torch.tensor(x_list[i].size()).float()
            g_dims = torch.tensor(g_list[i].size()).float()
            dist.send(x_dims,i)
            dist.send(g_dims,i)
        x_dims = torch.tensor(x_list[0].size())
        g_dims = torch.tensor(g_list[0].size())
    else:
        dist.recv(x_dims,0)
        dist.recv(g_dims,0)
    dist.barrier()

    # We get the actual interpolated tensors initialising placeholders with dims shared above
    x_data = torch.zeros(x_dims.int().tolist())
    g_data = torch.zeros(g_dims.int().tolist())
    if rank == 0:
        for i in range(1,size):
            x_data = x_list[i]
            g_data = g_list[i].float()
            dist.send(x_data,i)
            dist.send(g_data,i)
        x_data = x_list[0]
        g_data = g_list[0]
    else:
        dist.recv(x_data,0)
        dist.recv(g_data,0)
    dist.barrier()
    time_per_iter = 0

    # Create models on different devices
    if cuda_:
        torch.cuda.set_device(rank) ###set device to build stuff (GPU)
        model = Dist_ResNet_MNIST(Bottleneck,[3,300,300,3],type_=block_type).cuda() 
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        model = Dist_ResNet_MNIST(Bottleneck,[3,400,400,3],type_=block_type)
        criterion = torch.nn.CrossEntropyLoss()


# This part is for interpolating the states (weights) and distributing over devices

#     state = copy.deepcopy(model.state_dict())
#     for key in list(state):
#         new_key = key.replace(".","_")
#         state[new_key] = state.pop(key)
#         if rank == 0:
#             if new_key in weights:
#                 state[new_key] = weights[new_key]
#     globals().update(state)
#     dist.barrier()
#     for key in list(state):
#         dist.broadcast(eval(str(key)),0)
#     dist.barrier()
#     state_ = {k.replace("_","."): v for k, v in globals().items() if k.replace("_",".") in model.state_dict()}
#     model_state = model.state_dict()
#     model_state.update(state_)
#     model.load_state_dict(model_state)


    opt = torch.optim.Adam(model.parameters(),1e-3,weight_decay=1e-3) #change to adam
    t_1 = time.time()
    dt1 = time.time()-t_
    total_time += dt1
    t_time = time.time()
    c = 20
    iterand = 20 #30
    weight_reinit = 0
    for i in range(iterand): 
        for w in range(len(x_data)):
            t_ = time.time()
            input_init = x_data[w]
            if rank%2 == 0: #even and odd ranks to control send/recieve operations
                inp_1 = torch.tensor(input_init).float()
                if cuda_:
                    inp_1 = torch.tensor(input_init).float().cuda()
                if rank != size - 1:
                    grad_init = g_data[w]
                    grad_1 = torch.tensor(grad_init).float()
                    if cuda_:
                        grad_1 = torch.tensor(grad_init).float().cuda()
            else:
                inp_2 = torch.tensor(input_init).float()
                if cuda_:
                    inp_2 = torch.tensor(input_init).float().cuda()
                if rank != size - 1:
#                     grad_init = torch.cat(tuple(g_data),dim=0)
                    grad_init = g_data[w]
                    grad_2 = torch.tensor(grad_init).float()
                    if cuda_:
                        grad_2 = torch.tensor(grad_init).float().cuda()
            opt.zero_grad()
            dt1 = time.time()-t_
            total_time += dt1
            dist.barrier()
            if i==iterand-1:
                return
            for local_correction in range(c):
                t_2=time.time()
                t_ = time.time()
                if rank % 2 == 0:
                    if cuda_:
                        x = inp_1.data.cuda() #temp can kill later
                    else:
                        x = inp_1.data
                    x.requires_grad=True
                    inp_2 = model(x)
                    if rank==size-1:
                        arg_maxs = torch.argmax(inp_2.data, dim=1)
                        if cuda_:
                            loss = criterion(inp_2,g_data[w].long().cuda())
                            num_correct = torch.sum(g_data[w].long().cuda()==arg_maxs)
                        else:
                            loss = criterion(inp_2,g_data[w].long()) 
                            num_correct = torch.sum(g_data[w].long()==arg_maxs)
                        loss.backward()
                        acc = (num_correct * 100.0 / len(arg_maxs))
                        if local_correction >= 0:
                            print("Training",i,"LEpoch: ",local_correction,"Time per iter: ",time_per_iter,"GPU ID: ",rank,"Loss: ",loss.data, "Accuracy :",acc, " %")
                    grad_2 = x.grad
                else:
                    if cuda_:
                        x = inp_2.data.cuda() #temp can kill later
                    else:
                        x = inp_2.data
                    x.requires_grad=True
                    inp_1 = model(x)
                    if rank==size-1:
                        arg_maxs = torch.argmax(inp_1.data, dim=1)
                        if cuda_:
                            loss = criterion(inp_1,g_data[w].long().cuda())
                            num_correct = torch.sum(g_data[w].long().cuda()==arg_maxs)
                        else:
                            loss = criterion(inp_1,g_data[w].long()) 
                            num_correct = torch.sum(g_data[w].long()==arg_maxs)                        
                        loss.backward()
                        acc = (num_correct * 100.0 / len(arg_maxs))
                        if i>=0 and local_correction >= 0 and loss.data<0.01:
#                             print ("GPU",rank,"time: ",total_time, "loss: ", loss.data)
                            cust_val = torch.tensor(1)
#                             print("Training",i,"LEpoch: ",local_correction,"GPU ID: ",rank,"Loss: ",loss.data, "Accuracy :",acc, " %")
                    grad_1 = x.grad
                dist.broadcast(cust_val,1)
                if cust_val.data==1:
                    print("GPU: ",rank,"Time taken: ",total_time)
                    return
                opt.step()
                dt1 = time.time()-t_
                total_time += dt1
                t_ = time.time()


                # Send receive operations are done by even/odd 
                if rank==0:
                    dist.send(inp_2,dst=rank+1) #send x to next rank
                elif rank % 2 == 1:
                    dist.recv(inp_2,src=rank-1)
                    if rank!=size-1:
                        dist.send(inp_1,dst=rank+1) #send x to next rank
                elif rank % 2 == 0:
                    dist.recv(inp_1,src=rank-1)
                    if rank!=size-1:
                        dist.send(inp_2,dst=rank+1)
                if rank == 0:
                    dist.recv(grad_1,src=rank+1)
                elif rank % 2 == 1:
                    dist.send(grad_1,dst=rank-1)
                    if rank!=size-1:
                        dist.recv(grad_2,src=rank+1)
                elif rank % 2 == 0:
                    dist.send(grad_2,dst=rank-1)
                    if rank!=size-1:
                        dist.recv(grad_1,src=rank+1)
                if rank!=size-1:
                    if rank % 2 == 0:
                        if local_correction!=0:
                            inp_2.backward(grad_1)
                    else:
                        if local_correction!=0:
                            inp_1.backward(grad_2)
                opt.zero_grad()
                dt1 = time.time()-t_
                total_time += dt1
                dist.barrier()
                time_per_iter = time.time()-t_2
            t_ = time.time()
            if i<20:
                if rank%2 == 0:
                    if rank!=0:
                        x_data[w] = inp_1.data
                    if rank!= size-1:
                        g_data[w] = grad_1.data
                else:
                    if rank!=0:
                        x_data[w] = inp_2.data
                    if rank!=size-1:
                        g_data[w] = grad_2.data
            dt1 = time.time()-t_
            total_time += dt1
            t_ = time.time()

# option to restart weights 
#             if i == 0: #restart weights
#                 model_state = model.state_dict()
#                 model_state.update(state_)
#                 model.load_state_dict(model_state)
# #                 print("reloaded weights")
            dt1 = time.time()-t_
            total_time += dt1
            weight_reinit += dt1
#         if i == iterand-2:
#             print("GPU: ",rank,"Time taken: ",total_time)

def init_processes(fn,block_type):
    """ Initialize the distributed environment. """
    dist.init_process_group('mpi')
    rank = dist.get_rank()
    size = dist.get_world_size()
#     print('I am rank ', rank, ' on ', platform.node())
    cuda_ = True
    fn(rank,size,block_type[rank],cuda_)
    

if __name__=="__main__":
    block_type = [0,1,1,2]
#     block_type = [0,2] 
    init_processes(distribute_train,block_type)
# initialize()
