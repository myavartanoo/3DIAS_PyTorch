import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
from model.metric import chamfer_distance_naive, surfaceSampling, IOU, fscore
from model.loss import PI_value_generator, PI_funcs_generator
from utils.util import gen_polynomial_orders
import matplotlib.pyplot as plt

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    test_data_loader = config.init_obj('test_data_loader', module_data)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    checkname = str(config.run_id)
    os.makedirs('./visualization/'+checkname+"/functions/", exist_ok=True)
    os.makedirs('./visualization/'+checkname+"/valid_indices/", exist_ok=True)
    cls1_name = ['02691156', '02828884', '02933112', '02958343', '03001627', '03211117', '03636649', '03691459', '04090263', '04256520', '04379243', '04401088', '04530566']
    for i in range(len(cls1_name)):
        os.makedirs('./visualization/'+checkname+'/functions/'+cls1_name[i], exist_ok=True)
        os.makedirs('./visualization/'+checkname+'/valid_indices/'+cls1_name[i], exist_ok=True)
    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    sum_iou_cls = torch.zeros(13).to('cuda')
    sum_f_cls = torch.zeros(13).to('cuda')
    sum_chamfer_cls = torch.zeros(13).to('cuda')
    sum_iou_cls_mean = torch.zeros(13).to('cuda')
    sum_f_cls_mean = torch.zeros(13).to('cuda')
    sum_cham_cls_mean = torch.zeros(13).to('cuda')
    count_cls = torch.zeros(13).to('cuda')

    with torch.no_grad():
        polyorder_cpu = gen_polynomial_orders(4)
        for img_H, target in tqdm(test_data_loader):
            img_H = img_H.to('cuda')
            for key in target:
                if target[key].size==0: print(target['directory']); raise
                if key!='directory':
                    target[key] = target[key].to('cuda') 

            polycoeff, _, A_10x10 = model(img_H)

            polyorder = torch.from_numpy(gen_polynomial_orders(4)).to('cuda')
            PI_funcs_inout = PI_funcs_generator(target['inoutpts'], polycoeff, polyorder)
            PI_value_inout, _ = PI_value_generator(torch.tanh(PI_funcs_inout))

            batchiou = IOU(PI_value_inout, target)[:]
            batchf = fscore(PI_value_inout, target)[:]
            sampled_point = surfaceSampling(polycoeff, test_data_loader.dataset.allpoints) # on points
            batchchamferL1 = chamfer_distance_naive(sampled_point, target['onpts'])

            surfs, validinds = coeff2polystr(polycoeff.detach().cpu().numpy(), polyorder_cpu, A_10x10, PI_funcs_inout)

            for ii in range(batchiou.shape[0]):
                sum_iou_cls_mean[target['class_num'][ii]] += batchiou[ii]
                sum_f_cls_mean[target['class_num'][ii]] += batchf[ii]
                sum_cham_cls_mean[target['class_num'][ii]] += batchchamferL1[ii]
                if batchiou[ii] >= sum_iou_cls[target['class_num'][ii]]:
                    print('Check: ',batchiou[ii], target['directory'][ii])
                sum_iou_cls[target['class_num'][ii]] = max(batchiou[ii],sum_iou_cls[target['class_num'][ii]])
                sum_f_cls[target['class_num'][ii]] = max(batchiou[ii],sum_f_cls[target['class_num'][ii]])
                sum_chamfer_cls[target['class_num'][ii]] = max(batchiou[ii],sum_chamfer_cls[target['class_num'][ii]])
                count_cls[target['class_num'][ii]] += 1

                # save sample images, or do something with output here
                with open('./visualization/'+checkname+'/functions/'+'0'+str(target['directory'][ii][0].item())+'/'+str(target['directory'][ii][1].item())+'.txt', 'w') as f:
                        f.write(surfs[ii])
                with open('./visualization/'+checkname+'/valid_indices/'+'0'+str(target['directory'][ii][0].item())+'/'+str(target['directory'][ii][1].item())+'.txt', 'w') as f:
                        f.write(validinds[ii])


    cls_name = ['plane', 'bench', 'cabinet', 'car', 'chair', 'display', 'lamp', 'speaker', 'rifle', 'sofa', 'table', 'phone', 'vessel']
    iou_per_cls = sum_iou_cls_mean/count_cls
    f_per_cls = sum_f_cls_mean/count_cls
    cham_per_cls = sum_cham_cls_mean/count_cls
    print("class names:", cls_name)
    print("# samples  :", count_cls)
    for ii in range(13):
        print(cls_name[ii]+" IoU: {}".format(iou_per_cls[ii]))
        print(cls_name[ii]+" chamfer: {}".format(cham_per_cls[ii]))
        print(cls_name[ii]+" F: {}".format(f_per_cls[ii]))
    print("#####################")
    print("naive average IoU: {}".format(torch.mean(iou_per_cls).item()))
    print("Total average IoU: {}".format(torch.sum(sum_iou_cls_mean)/torch.sum(count_cls)))
    print("naive average F: {}".format(torch.mean(f_per_cls).item()))
    print("Total average F: {}".format(torch.sum(sum_f_cls_mean)/torch.sum(count_cls)))
    print("naive average chamfer: {}".format(torch.mean(cham_per_cls).item()))
    print("Total average chamfer: {}".format(torch.sum(sum_cham_cls_mean)/torch.sum(count_cls)))


def coeff2polystr(polycoeff, polyorder, A_10x10, PI_funcs_inout):
    """
    polycoeff (=Params): 
        (batch, num_params, num_functions) = (batch, 35, 32)
    polyorders: 
        (num_params, 3) = (35, 3) - degree of x,y,z for each term in polynomial.
        Since we use 4th-polynomials, the number of parameters is determined by (4+1)(4+1+1)(4+1+2)/6 = 35
    """

    batchsurfs = ["" for i in range(polycoeff.shape[0])]
    batchind = ["" for i in range(polycoeff.shape[0])]
    
    for batch in range(polycoeff.shape[0]): #batch
        surface = ""
        c = 0
        list_valid = []
        for subf in range(polycoeff.shape[2]): #100 = 25*4
            f_tmp = ""
            A_10x10[batch,:,:,subf] = A_10x10[batch,:,:,subf]/torch.norm(A_10x10[batch,:,:,subf])
            if torch.min(torch.eig(A_10x10[batch,:,:,subf])[0][:,0])<-0.0 and torch.max(torch.eig(A_10x10[batch,:,:,subf])[0][:,0])>0 and torch.sum(torch.abs(torch.prod(torch.eig(A_10x10[batch,:,:,subf])[0][:,1])))==0 and torch.min(PI_funcs_inout[batch,:,subf])<0:        
                list_valid.append(subf)

                for term in range(polycoeff.shape[1]): #35
                    f_tmp = f_tmp + '+('+str(polycoeff[batch, term, subf])+')*(x^'+str(int(polyorder[term,0]))+')*(y^'+str(int(polyorder[term,1]))+')*(z^'+str(int(polyorder[term,2]))+')'
                if c==0:
                   surface = surface + f_tmp[1:]+'\n'
                else:
                    surface = 'min(' + surface + ',' + f_tmp[1:]+')'+'\n'
                c+=1
        valids = ""
        for sub_valid in list_valid:
            valids = valids + str(sub_valid) + ' '
        batchind[batch] = batchind[batch]  + valids #+ '-' + str(R[batch,0]) 
        batchsurfs[batch] = batchsurfs[batch]  + surface #+ '-' + str(R[batch,0]) 

    return batchsurfs, batchind


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DIAS_PyTorch')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    parser.add_argument('-t', '--tag', default=None, type=str,
                      help='experience name in tensorboard (default: None)')

    config = ConfigParser.from_args(parser.parse_args()) #, options)

    main(config)
