import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import gen_polynomial_orders

mse = nn.MSELoss(reduction='mean')
CELoss = nn.CrossEntropyLoss()


def PI_funcs_generator(points, coeff, polyorder):
    I = (points.unsqueeze(dim=2) ** polyorder).prod(dim=3) # (batch, num_points, num_params) - Calculating the order of x^a*y^b*z^c
    PI_funcs = (coeff.unsqueeze(dim=1) * I.unsqueeze(dim=3)).sum(dim=2) # (batch, num_points, num_functions)
    return PI_funcs # (batch, num_points, num_functions)

def PI_value_generator(PI_funcs):   
    union = torch.min(PI_funcs[:,:,:],dim=2)
    return union[0].unsqueeze(dim=2), union[1]

def PI_difffuncs_generator(points, coeff, polyorder):
    # Making [df1, df2, ... ] / [dx, dy, dz] - (batch, num_points, num_functions, 3)
    dcoeff = coeff.unsqueeze(dim=3) * polyorder.unsqueeze(dim=1).unsqueeze(dim=0)  # (batch, num_params, num_functions, 3)

    dpolyorder = polyorder.unsqueeze(dim=2)
    dpolyorder = torch.cat((dpolyorder, dpolyorder, dpolyorder), dim=2)  # (num_params, 3 for x/y/z, 3 for dx/dy/dz) (e.g. dpolyorder[:,:,0] contains orders differentiated by x)
    dpolyorder = dpolyorder - torch.eye(3, device='cuda', dtype=torch.float32)
    dpolyorder = torch.max(dpolyorder, torch.zeros(1, device='cuda', dtype=torch.float32))  # avoid 0 ** -1
    dI = (points.unsqueeze(dim=2).unsqueeze(dim=4) ** dpolyorder.unsqueeze(dim=0).unsqueeze(dim=1)).prod(dim=3) # (batch, num_points, num_params, 3)

    PI_diff = (dcoeff.unsqueeze(dim=1) * dI.unsqueeze(dim=3)).sum(dim=2)  # (batch, num_points, num_functions, 3)
    return PI_diff

def PI_diff_generator_torch(PI_diff, min_idx):
    idx = min_idx
    idx = torch.cat([idx.unsqueeze(-1).unsqueeze(-1),idx.unsqueeze(-1).unsqueeze(-1),idx.unsqueeze(-1).unsqueeze(-1)],dim=-1)

    Diff_funcs =  torch.gather(PI_diff, 2, idx)[:,:,0,:]
    PI_normal = torch.nn.functional.normalize(Diff_funcs, p=2, dim=2)  # normalized vectors / negative direction
    return PI_normal

def loss_points(union_on, union_inout, target, loss_weights):
    num_inside = target['numinside']

    loss_on = mse(union_on, torch.zeros_like(union_on)) 
    loss_out = 0
    loss_in = 0
    for i in range(num_inside.shape[0]):
        loss_in  += mse(union_inout[i,:num_inside[i]],-torch.ones_like(union_inout[i,:num_inside[i]])) # inside
        loss_out += mse(union_inout[i,num_inside[i]:], torch.ones_like(union_inout[i,num_inside[i]:])) # outside
    loss_in = loss_in/(num_inside.shape[0])
    loss_out = loss_out/(num_inside.shape[0])

    loss = loss_weights['on']*loss_on + loss_weights['in']*loss_in + loss_weights['out']*loss_out

    return loss_on, loss_in, loss_out, loss

def loss_normvec(target, normal):
    """ target: (batch, num_point_on, 3) - true normal vector at the position of an on point """
    loss_normvec = mse(normal, target)
    if not torch.isfinite(loss_normvec): 
        print("problem on loss_normvec\n", "normal\n",normal,"\nnormal SHAPE\n",normal.shape); 
        for batches in range(normal.shape[0]):
            if not torch.isfinite(mse(normal[batches], target[batches])):
                print("idx:",batches)
        raise
    return loss_normvec

def total_loss(polycoeff, target, loss_weights):
    ## Point loss 
    polyorder = torch.from_numpy(gen_polynomial_orders(4)).to('cuda')
    PI_funcs_on = PI_funcs_generator(target['onpts'], polycoeff, polyorder) # (batch, num_points, num_functions)
    PI_funcs_inout = PI_funcs_generator(target['inoutpts'], polycoeff, polyorder) # (batch, num_points, num_functions)
    
    union_on, min_idx_on = PI_value_generator(torch.tanh(PI_funcs_on)) # (batch, num_points, 1)
    union_inout, _ = PI_value_generator(torch.tanh(PI_funcs_inout)) # (batch, num_points, 1)

    lossPTSon, lossPTSin, lossPTSout, lossPTS = loss_points(union_on, union_inout, target, loss_weights)

    ## Normal vector loss
    PI_diff = PI_difffuncs_generator(target['onpts'], polycoeff, polyorder) # Making [df1, df2, ... ] / [dx, dy, dz] - (batch, num_points, num_functions, 3)
    normal = PI_diff_generator_torch(PI_diff, min_idx_on)
    lossNormVec = loss_weights['normvec']*loss_normvec(target['normal'], normal)

    return lossPTS + lossNormVec,\
            {'loss_pnt_on': lossPTSon.detach().item(), 
             'loss_pnt_in': lossPTSin.detach().item(), 
             'loss_pnt_out':lossPTSout.detach().item(),
             'loss_normvec':lossNormVec.detach().item()
             }, union_inout.detach()