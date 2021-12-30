import torch
import numpy as np
from model.loss import PI_value_generator, PI_funcs_generator   
from utils.util import gen_polynomial_orders

def IOU(PI_value_inout, target, per_class=True):
    # per batch
    ininA  = torch.zeros_like(target['numinside'])
    inoutB = torch.zeros_like(target['numinside'])
    outinC = torch.zeros_like(target['numinside'])
    for i in range(target['numinside'].shape[0]):
        ininA[i]  = torch.sum(PI_value_inout[i,:target['numinside'][i] ]<=0, dim=0) # (batch)
        inoutB[i] = torch.sum(PI_value_inout[i,:target['numinside'][i] ]>0, dim=0) # estimated outside but actually inside
        outinC[i] = torch.sum(PI_value_inout[i, target['numinside'][i]:]<=0, dim=0) # (batch)

    if per_class:
        return torch.true_divide(ininA, ininA + inoutB + outinC)  # IOU per batch
    else:
        return torch.mean(torch.true_divide(ininA, ininA + inoutB + outinC))


def fscore(PI_value_inout, target, per_class=True):
    # per batch
    ininA  = torch.zeros_like(target['numinside'])
    inoutB = torch.zeros_like(target['numinside'])
    outinC = torch.zeros_like(target['numinside'])
    for i in range(target['numinside'].shape[0]):
        ininA[i]  = torch.sum(PI_value_inout[i,:target['numinside'][i] ]<=0, dim=0) # (batch)
        inoutB[i] = torch.sum(PI_value_inout[i,:target['numinside'][i] ]>0, dim=0) # estimated outside but actually inside
        outinC[i] = torch.sum(PI_value_inout[i, target['numinside'][i]:]<=0, dim=0) # (batch)
    
    precision = torch.zeros(ininA.shape, device='cuda')
    vidx_pre = ininA+outinC > 0
    precision[vidx_pre] = torch.true_divide(ininA[vidx_pre], (ininA+outinC)[vidx_pre])

    recall = torch.zeros(ininA.shape, device='cuda')
    vidx_rec = ininA+inoutB > 0
    recall[vidx_rec] = torch.true_divide(ininA[vidx_rec], (ininA+inoutB)[vidx_rec])
          
    F = torch.zeros(ininA.shape, device='cuda')
    vidx_f =  precision+recall > 0
    F[vidx_f] = 2*torch.true_divide((precision*recall)[vidx_f], (precision+recall)[vidx_f])

    if per_class:
        return F  # IOU per batch
    else:
        return torch.mean(F)


def surfaceSampling(coeff, allpoints):
    polyorder = torch.from_numpy(gen_polynomial_orders(4)).to('cuda')
    allpoints = torch.from_numpy(allpoints).unsqueeze(0).to('cuda')

    PI_funcs = PI_funcs_generator(allpoints, coeff, polyorder)
    PI_value, _ = PI_value_generator(torch.tanh(PI_funcs))

    idx = torch.argsort(torch.abs(PI_value),dim=1)[:,:2000][0,:,0]
    return allpoints[:,idx]


def chamfer_distance_naive(points1, points2):
    ''' Naive implementation of the Chamfer distance.
    Args:
        points1 (batch, num_on_points, 3)
        points2 (batch, num_on_points, 3)
    '''
    assert(points1.shape == points2.shape)
    batch_size, T, _ = points1.shape

    points1 = points1.view(batch_size, T, 1, 3)
    points2 = points2.view(batch_size, 1, T, 3)

    distances = torch.abs(points1 - points2).sum(-1)

    chamfer1 = distances.min(dim=1)[0].mean(dim=1)
    chamfer2 = distances.min(dim=2)[0].mean(dim=1)

    chamfer = chamfer1 + chamfer2
    return chamfer


