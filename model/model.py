import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model.ResNet import resnet18

class Shape3DModel(BaseModel):
    def __init__(self, num_param, num_functions, num_classes):
        super().__init__()
        self.num_param = num_param
        self.num_functions = num_functions
        self.num_classes = num_classes
        self.num_coeff = 35
        self.num_function = [100]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # model: resnet_batchnorm2
        self.gen_r = nn.Linear(num_param*num_functions, 1)
        self.gen_lambda = nn.Linear(num_param*num_functions, 1)
        # encoder
        self.resnet_h = resnet18(num_classes=1000, pretrained=True)

        # results
        self.gen_paramsh_0 = nn.Linear(1000, 1024)
        self.gen_paramsh_1 = nn.Linear(1000, 4096)
        self.gen_paramsh_2 = nn.Linear(4096, 4096)
        self.gen_paramsh_3 = nn.Linear(4096, 4096)
        self.gen_paramsh_4 = nn.Linear(4096, num_param*self.num_function[0])

        self.orgh_0 = nn.Linear(1000, 1024)
        self.orgh_1 = nn.Linear(1000, 1024)
        self.orgh_2 = nn.Linear(1024, 512)
        self.orgh_3 = nn.Linear(512, 256)
        self.orgh_4 = nn.Linear(256, self.num_function[0]*3)

        self.fc1 = nn.Linear(35*num_functions , 1000)
        self.fc2 = nn.Linear(1000, num_classes)
        
        self.fc_pd1 = nn.Linear(1000, 256)
        self.fc_pd2 = nn.Linear(256, self.num_function[0])

        self.batchnorm_h_0 = nn.BatchNorm1d(1024)
        self.batchnorm_h_1 = nn.BatchNorm1d(4096)
        self.batchnorm_h_2 = nn.BatchNorm1d(4096)
        self.batchnorm_h_3 = nn.BatchNorm1d(4096)
        
        self.batchnorm_org_h_0 = nn.BatchNorm1d(1024)
        self.batchnorm_org_h_1 = nn.BatchNorm1d(1024)
        self.batchnorm_org_h_2 = nn.BatchNorm1d(512)
        self.batchnorm_org_h_3 = nn.BatchNorm1d(256)
        
        self.drop0 = nn.Dropout(p=0.0)
        self.drop1 = nn.Dropout(p=0.0)
        self.drop2 = nn.Dropout(p=0.0)
        self.drop3 = nn.Dropout(p=0.0)
            

    def forward(self,img_H):

        def _pd_full(netcoeff, R):  
            '''
            netcoeff: (batch, 55, num_functions)
            R : (batch, num_functions)
            '''
            #### generate symm-matrix B
            B = torch.zeros((netcoeff.shape[0]*netcoeff.shape[2], 10, 10), device=self.device)
            triu_idcs = torch.triu_indices(row=10, col=10, offset=0).to(self.device)
            B[:, triu_idcs[0], triu_idcs[1]] = netcoeff.reshape(-1,55)  # vector to upper triangular matrix
            B[:, triu_idcs[1], triu_idcs[0]] = netcoeff.reshape(-1,55)  # B: symm. matrix

            #### generate A_10x10 from symm-matrix B
            A = torch.bmm(B, B)  # A = B**2  // A: symm. positive definite (batch*num_funcs, 6,6)
            A.add_(torch.eye(10, device=self.device)/10000) # for stability
            A = A.reshape(netcoeff.shape[0], netcoeff.shape[2], 10, 10)  # (batch, num_funcs, 6, 6)
            A = A.permute(0, 2, 3, 1)  # (batch, 6, 6, num_funcs)
            
            #### add boundary R 
            A[:,0,0,:] = A[:,0,0,:] - R  
            A[:,4,4,:] = A[:,4,4,:] + 1  
            A[:,5,5,:] = A[:,5,5,:] + 1  
            A[:,6,6,:] = A[:,6,6,:] + 1 

            #### generate polynomial coefficent from matrix A_10x10
            # order of v: [1, z, y, x, x^2, y^2, z^2, xy, yz, xz] -> vAv'
            # order of pcoeff: [1, z, z^2, z^3, y, yz, yz^2, y^2, y^2z, y^3, x, xz, xz^2, xy, xyz, xy^2, x^2, x^2z, x^2y, x^3,
            #                   z^4, yz^3, y^2z^2, y^3z, y^4, xz^3, xyz^2, xy^2z, xy^3, x^2z^2, x^2yz, x^2y^2, x^3z, x^3y, x^4]
            pcoeff = torch.zeros((netcoeff.shape[0], 35, netcoeff.shape[2]), device=self.device)
            pcoeff[:,0,:] = A[:,0,0,:]  # 1
            pcoeff[:,1,:] = A[:,0,1,:]+A[:,1,0,:] # z
            pcoeff[:,2,:] = A[:,0,6,:]+A[:,6,0,:]+A[:,1,1,:] # z^2
            pcoeff[:,3,:] = A[:,1,6,:]+A[:,6,1,:] # z^3
            pcoeff[:,4,:] = A[:,0,2,:]+A[:,2,0,:] # y
            pcoeff[:,5,:] = A[:,1,2,:]+A[:,2,1,:]+A[:,0,8,:]+A[:,8,0,:] # yz
            pcoeff[:,6,:] = A[:,2,6,:]+A[:,6,2,:]+A[:,1,8,:]+A[:,8,1,:] # yz^2
            pcoeff[:,7,:] = A[:,0,5,:]+A[:,5,0,:]+A[:,2,2,:] # y^2
            pcoeff[:,8,:] = A[:,1,5,:]+A[:,5,1,:]+A[:,2,8,:]+A[:,8,2,:] # y^2z
            pcoeff[:,9,:] = A[:,2,5,:]+A[:,5,2,:] # y^3
            pcoeff[:,10,:] = A[:,0,3,:]+A[:,3,0,:] # x
            pcoeff[:,11,:] = A[:,1,3,:]+A[:,3,1,:]+A[:,0,9,:]+A[:,9,0,:] # xz
            pcoeff[:,12,:] = A[:,3,6,:]+A[:,6,3,:]+A[:,1,9,:]+A[:,9,1,:] # xz^2
            pcoeff[:,13,:] = A[:,2,3,:]+A[:,3,2,:]+A[:,0,7,:]+A[:,7,0,:] # xy
            pcoeff[:,14,:] = A[:,1,7,:]+A[:,7,1,:]+A[:,2,9,:]+A[:,9,2,:]+A[:,3,8,:]+A[:,8,3,:] # xyz
            pcoeff[:,15,:] = A[:,3,5,:]+A[:,5,3,:]+A[:,2,7,:]+A[:,7,2,:] # xy^2
            pcoeff[:,16,:] = A[:,0,4,:]+A[:,4,0,:]+A[:,3,3,:] # x^2
            pcoeff[:,17,:] = A[:,1,4,:]+A[:,4,1,:]+A[:,3,9,:]+A[:,9,3,:] # x^2z
            pcoeff[:,18,:] = A[:,2,4,:]+A[:,4,2,:]+A[:,3,7,:]+A[:,7,3,:] # x^2y
            pcoeff[:,19,:] = A[:,3,4,:]+A[:,4,3,:] # x^3
            pcoeff[:,20,:] = A[:,6,6,:] # z^4
            pcoeff[:,21,:] = A[:,6,8,:]+A[:,8,6,:] # yz^3
            pcoeff[:,22,:] = A[:,5,6,:]+A[:,6,5,:]+A[:,8,8,:] # y^2z^2
            pcoeff[:,23,:] = A[:,5,8,:]+A[:,8,5,:] # y^3z
            pcoeff[:,24,:] = A[:,5,5,:] # y^4
            pcoeff[:,25,:] = A[:,6,9,:]+A[:,9,6,:] # xz^3
            pcoeff[:,26,:] = A[:,6,7,:]+A[:,7,6,:]+A[:,8,9,:]+A[:,9,8,:] # xyz^2
            pcoeff[:,27,:] = A[:,5,9,:]+A[:,9,5,:]+A[:,7,8,:]+A[:,8,7,:] # xy^2z
            pcoeff[:,28,:] = A[:,5,7,:]+A[:,7,5,:] # xy^3
            pcoeff[:,29,:] = A[:,4,6,:]+A[:,6,4,:]+A[:,9,9,:] # x^2z^2
            pcoeff[:,30,:] = A[:,4,8,:]+A[:,8,4,:]+A[:,7,9,:]+A[:,9,7,:] # x^2yz
            pcoeff[:,31,:] = A[:,4,5,:]+A[:,5,4,:]+A[:,7,7,:] # x^2y^2
            pcoeff[:,32,:] = A[:,4,9,:]+A[:,9,4,:] # x^3z
            pcoeff[:,33,:] = A[:,4,7,:]+A[:,7,4,:] # x^3y
            pcoeff[:,34,:] = A[:,4,4,:] # x^4

            return pcoeff, A         

        def _gen_polycoeff_center(net_params, origins):
            '''
            net_params: (batch, 35, num_functions)
            origins : (batch, 3, num_functions)
            '''
            polycoeff_center = torch.zeros_like(net_params)
            polycoeff_center[:,0,:] = net_params[:,0,:]-net_params[:,1,:]*origins[:,2,:]+net_params[:,2,:]*(origins[:,2,:]**2)-net_params[:,3,:]*(origins[:,2,:]**3)-net_params[:,4,:]*(origins[:,1,:])+net_params[:,5,:]*(origins[:,2,:])*(origins[:,1,:]) \
                                - net_params[:, 6, :]*(origins[:,2,:]**2)*(origins[:,1,:]) +net_params[:,7,:]*(origins[:,1,:]**2)-net_params[:,8,:]*(origins[:,2,:])*(origins[:,1,:]**2)-net_params[:,9,:]*(origins[:,1,:]**3)-net_params[:,10,:]*(origins[:,0,:])+net_params[:,11,:]*(origins[:,0,:])*(origins[:,2,:])\
                                -net_params[:,12,:]*(origins[:,0,:])*(origins[:,2,:]**2)+net_params[:,13,:]*(origins[:,0,:])*(origins[:,1,:])-net_params[:,14,:]*(origins[:,0,:])*(origins[:,1,:])*(origins[:,2,:])-net_params[:,15,:]*(origins[:,0,:])*(origins[:,1,:]**2)+net_params[:,16,:]*(origins[:,0,:]**2)-net_params[:,17,:]*(origins[:,0,:]**2)*(origins[:,2,:])\
                                -net_params[:,18,:]*(origins[:,0,:]**2)*(origins[:,1,:])-net_params[:,19,:]*(origins[:,0,:]**3)+net_params[:,20,:]*(origins[:,2,:]**4)+net_params[:,21,:]*(origins[:,2,:]**3)*(origins[:,1,:])+net_params[:,22,:]*(origins[:,2,:]**2)*(origins[:,1,:]**2)+net_params[:,23,:]*(origins[:,1,:]**3)*(origins[:,2,:])+net_params[:,24,:]*(origins[:,1,:]**4)\
                                +net_params[:,25,:]*(origins[:,2,:]**3)*(origins[:,0,:])+net_params[:,26,:]*(origins[:,1,:])*(origins[:,2,:]**2)*(origins[:,0,:])+net_params[:,27,:]*(origins[:,1,:]**2)*(origins[:,2,:])*(origins[:,0,:])+net_params[:,28,:]*(origins[:,0,:])*(origins[:,1,:]**3)+net_params[:,29,:]*(origins[:,0,:]**2)*(origins[:,2,:]**2)+net_params[:,30,:]*(origins[:,0,:]**2)*(origins[:,1,:])*(origins[:,2,:])\
                                +net_params[:,31,:]*(origins[:,0,:]**2)*(origins[:,1,:]**2)+net_params[:,32,:]*(origins[:,0,:]**3)*(origins[:,2,:])+net_params[:,33,:]*(origins[:,0,:]**3)*(origins[:,1,:])+net_params[:,34,:]*(origins[:,0,:]**4)
            polycoeff_center[:, 1, :] = net_params[:, 1, :] - 2*net_params[:, 2, :]*(origins[:,2,:]) + 3*net_params[:, 3, :]*(origins[:,2,:]**2) - net_params[:, 5, :]*(origins[:,1,:]) +2* net_params[:, 6, :]*(origins[:,2,:])*(origins[:,1,:])\
                                + net_params[:, 8, :]*(origins[:,1,:]**2) - net_params[:, 11, :]*(origins[:,0,:]) + 2*net_params[:, 12, :]*(origins[:,0,:])*(origins[:,2,:]) + net_params[:, 14, :]*(origins[:,0,:])*(origins[:,1,:]) + net_params[:, 17, :]*(origins[:,0,:]**2)\
                                - 4*net_params[:, 20, :]*(origins[:,2,:]**3) - 3*net_params[:, 21, :]*(origins[:,2,:]**2)*(origins[:,1,:]) - 2*net_params[:, 22, :]*(origins[:,2,:])*(origins[:,1,:]**2) - net_params[:, 23, :]*(origins[:,1,:]**3) - 3*net_params[:, 25, :]*(origins[:,2,:]**2)*(origins[:,0,:])\
                                - 2*net_params[:, 26, :]*(origins[:,1,:])*(origins[:,2,:])*(origins[:,0,:]) - net_params[:, 27, :]*(origins[:,1,:]**2)*(origins[:,0,:]) - net_params[:, 30, :]*(origins[:,0,:]**2)*(origins[:,1,:]) - net_params[:, 32, :]*(origins[:,0,:]**3) - 2*net_params[:, 29, :]*(origins[:,0,:]**2)*(origins[:,2,:])
            polycoeff_center[:, 2, :] = net_params[:, 2, :] - 3*net_params[:, 3, :]*(origins[:,2,:]) - net_params[:, 6, :]*(origins[:,1,:]) - net_params[:, 12, :]*(origins[:,0,:]) + 6*net_params[:, 20, :]*(origins[:,2,:]**2)\
                                + 3*net_params[:, 21, :]*(origins[:,2,:])*(origins[:,1,:]) + net_params[:, 22, :]*(origins[:,1,:]**2) + 3*net_params[:, 25, :]*(origins[:,2,:])*(origins[:,0,:]) + net_params[:, 26, :]*(origins[:,1,:])*(origins[:,0,:]) + net_params[:, 29, :]*(origins[:,0,:]**2)
            polycoeff_center[:, 3, :] = net_params[:, 3, :] - 4*net_params[:, 20, :]*(origins[:,2,:]) - net_params[:, 21, :]*(origins[:,1,:]) - net_params[:, 25, :]*(origins[:,0,:])
            polycoeff_center[:, 4, :] = net_params[:, 4, :] - net_params[:, 5, :]*(origins[:,2,:]) + net_params[:, 6, :]*(origins[:,2,:]**2) - 2*net_params[:, 7, :]*(origins[:,1,:]) +2* net_params[:, 8, :]*(origins[:,2,:])*(origins[:,1,:])\
                                + 3*net_params[:, 9, :]*(origins[:,1,:]**2) - net_params[:, 13, :]*(origins[:,0,:]) + net_params[:, 14, :]*(origins[:,0,:])*(origins[:,2,:]) + 2*net_params[:, 15, :]*(origins[:,0,:])*(origins[:,1,:]) + net_params[:, 18, :]*(origins[:,0,:]**2)\
                                - net_params[:, 21, :]*(origins[:,2,:]**3) - 2*net_params[:, 22, :]*(origins[:,2,:]**2)*(origins[:,1,:]) - 3*net_params[:, 23, :]*(origins[:,2,:])*(origins[:,1,:]**2) - 4*net_params[:, 24, :]*(origins[:,1,:]**3) - net_params[:, 26, :]*(origins[:,2,:]**2)*(origins[:,0,:])\
                                - 2*net_params[:, 27, :]*(origins[:,1,:])*(origins[:,2,:])*(origins[:,0,:]) - 3*net_params[:, 28, :]*(origins[:,1,:]**2)*(origins[:,0,:]) - net_params[:, 30, :]*(origins[:,0,:]**2)*(origins[:,2,:]) - net_params[:, 33, :]*(origins[:,0,:]**3) - 2*net_params[:, 31, :]*(origins[:,0,:]**2)*(origins[:,1,:])
            polycoeff_center[:, 5, :] = net_params[:, 5, :] - 2*net_params[:, 6, :]*(origins[:,2,:]) - 2*net_params[:, 8, :]*(origins[:,1,:]) - net_params[:, 14, :]*(origins[:,0,:]) + 3*net_params[:, 21, :]*(origins[:,2,:]**2)\
                                + 4*net_params[:, 22, :]*(origins[:,2,:])*(origins[:,1,:]) + 3*net_params[:, 23, :]*(origins[:,1,:]**2) + 2*net_params[:, 26, :]*(origins[:,2,:])*(origins[:,0,:]) + 2*net_params[:, 27, :]*(origins[:,1,:])*(origins[:,0,:]) + net_params[:, 30, :]*(origins[:,0,:]**2)
            polycoeff_center[:, 6, :] = net_params[:, 6, :] - 3*net_params[:, 21, :]*(origins[:,2,:]) - 2*net_params[:, 22, :]*(origins[:,1,:]) - net_params[:, 26, :]*(origins[:,0,:])
            polycoeff_center[:, 7, :] = net_params[:, 7, :] - net_params[:, 8, :]*(origins[:,2,:]) - 3*net_params[:, 9, :]*(origins[:,1,:]) - net_params[:, 15, :]*(origins[:,0,:]) + net_params[:, 22, :]*(origins[:,2,:]**2)\
                                + 3*net_params[:, 23, :]*(origins[:,2,:])*(origins[:,1,:]) + 6*net_params[:, 24, :]*(origins[:,1,:]**2) + net_params[:, 27, :]*(origins[:,2,:])*(origins[:,0,:]) + 3*net_params[:, 28, :]*(origins[:,1,:])*(origins[:,0,:]) + net_params[:, 31, :]*(origins[:,0,:]**2)
            polycoeff_center[:, 8, :] = net_params[:, 8, :] - 2*net_params[:, 22, :]*(origins[:,2,:]) - 3*net_params[:, 23, :]*(origins[:,1,:]) - net_params[:, 27, :]*(origins[:,0,:])
            polycoeff_center[:, 9, :] = net_params[:, 9, :] - net_params[:, 23, :]*(origins[:,2,:]) - 4*net_params[:, 24, :]*(origins[:,1,:]) - net_params[:, 28, :]*(origins[:,0,:])
            polycoeff_center[:, 10, :] = net_params[:, 10, :] - net_params[:, 11, :]*(origins[:,2,:]) + net_params[:, 12, :]*(origins[:,2,:]**2) - net_params[:, 13, :]*(origins[:,1,:]) + net_params[:, 14, :]*(origins[:,2,:])*(origins[:,1,:])\
                                + net_params[:, 15, :]*(origins[:,1,:]**2) - 2*net_params[:, 16, :]*(origins[:,0,:]) + 2*net_params[:, 17, :]*(origins[:,0,:])*(origins[:,2,:]) + 2*net_params[:, 18, :]*(origins[:,0,:])*(origins[:,1,:]) + 3*net_params[:, 19, :]*(origins[:,0,:]**2)\
                                - net_params[:, 25, :]*(origins[:,2,:]**3) - net_params[:, 26, :]*(origins[:,2,:]**2)*(origins[:,1,:]) - net_params[:, 27, :]*(origins[:,2,:])*(origins[:,1,:]**2) - net_params[:, 28, :]*(origins[:,1,:]**3) - 2*net_params[:, 29, :]*(origins[:,2,:]**2)*(origins[:,0,:])\
                                - 2*net_params[:, 30, :]*(origins[:,1,:])*(origins[:,2,:])*(origins[:,0,:]) - 2*net_params[:, 31, :]*(origins[:,1,:]**2)*(origins[:,0,:]) - 3*net_params[:, 32, :]*(origins[:,0,:]**2)*(origins[:,2,:]) - 4*net_params[:, 34, :]*(origins[:,0,:]**3) - 3*net_params[:, 33, :]*(origins[:,0,:]**2)*(origins[:,1,:])
            polycoeff_center[:, 11, :] = net_params[:, 11, :] - 2*net_params[:, 12, :]*(origins[:,2,:]) - net_params[:, 14, :]*(origins[:,1,:]) - 2*net_params[:, 17, :]*(origins[:,0,:]) + 3*net_params[:, 25, :]*(origins[:,2,:]**2)\
                                + 2*net_params[:, 26, :]*(origins[:,2,:])*(origins[:,1,:]) + net_params[:, 27, :]*(origins[:,1,:]**2) + 4*net_params[:, 29, :]*(origins[:,2,:])*(origins[:,0,:]) + 2*net_params[:, 30, :]*(origins[:,1,:])*(origins[:,0,:]) + 3*net_params[:, 32, :]*(origins[:,0,:]**2)
            polycoeff_center[:, 12, :] = net_params[:, 12, :] - 3*net_params[:, 25, :]*(origins[:,2,:]) - net_params[:, 26, :]*(origins[:,1,:]) - 2*net_params[:, 29, :]*(origins[:,0,:])
            polycoeff_center[:, 13, :] = net_params[:, 13, :] - net_params[:, 14, :]*(origins[:,2,:]) - 2*net_params[:, 15, :]*(origins[:,1,:]) - 2*net_params[:, 18, :]*(origins[:,0,:]) + net_params[:, 26, :]*(origins[:,2,:]**2)\
                                + 2*net_params[:, 27, :]*(origins[:,2,:])*(origins[:,1,:]) + 3*net_params[:, 28, :]*(origins[:,1,:]**2) + 2*net_params[:, 30, :]*(origins[:,2,:])*(origins[:,0,:]) + 4*net_params[:, 31, :]*(origins[:,1,:])*(origins[:,0,:]) + 3*net_params[:, 33, :]*(origins[:,0,:]**2)
            polycoeff_center[:, 14, :] = net_params[:, 14, :] - 2*net_params[:, 26, :]*(origins[:,2,:]) - 2*net_params[:, 27, :]*(origins[:,1,:]) - 2*net_params[:, 30, :]*(origins[:,0,:])
            polycoeff_center[:, 15, :] = net_params[:, 15, :] - net_params[:, 27, :]*(origins[:,2,:]) - 3*net_params[:, 28, :]*(origins[:,1,:]) - 2*net_params[:, 31, :]*(origins[:,0,:])
            polycoeff_center[:, 16, :] = net_params[:, 16, :] - net_params[:, 17, :]*(origins[:,2,:]) - net_params[:, 18, :]*(origins[:,1,:]) - 3*net_params[:, 19, :]*(origins[:,0,:]) + net_params[:, 29, :]*(origins[:,2,:]**2)\
                                + net_params[:, 30, :]*(origins[:,2,:])*(origins[:,1,:]) + net_params[:, 31, :]*(origins[:,1,:]**2) + 3*net_params[:, 32, :]*(origins[:,2,:])*(origins[:,0,:]) + 3*net_params[:, 33, :]*(origins[:,1,:])*(origins[:,0,:]) + 6*net_params[:, 34, :]*(origins[:,0,:]**2)
            polycoeff_center[:, 17, :] = net_params[:, 17, :] - 1*net_params[:, 29, :]*(origins[:,2,:]) - net_params[:, 30, :]*(origins[:,1,:]) - 3*net_params[:, 32, :]*(origins[:,0,:])
            polycoeff_center[:, 18, :] = net_params[:, 18, :] - net_params[:, 30, :]*(origins[:,2,:]) - 2*net_params[:, 31, :]*(origins[:,1,:]) - 3*net_params[:, 33, :]*(origins[:,0,:])
            polycoeff_center[:, 19, :] = net_params[:, 19, :] - net_params[:, 32, :]*(origins[:,2,:]) - net_params[:, 33, :]*(origins[:,1,:]) - 4*net_params[:, 34, :]*(origins[:,0,:])
            polycoeff_center[:, 20:, :] = net_params[:, 20:, :] # origin does not effect on the highest term

            return polycoeff_center

        #### Format: (N, C, H, W)
        feature_resnet, _, _, _, _ = self.resnet_h(img_H)

        #### generate the elements for symm-matrix B
        net_params_B = self.drop1(F.relu(self.batchnorm_h_1(self.gen_paramsh_1(feature_resnet))))
        net_params_B = self.drop2(F.relu(self.batchnorm_h_2(self.gen_paramsh_2(net_params_B))))
        net_params_B = self.drop3(F.relu(self.batchnorm_h_3(self.gen_paramsh_3(net_params_B))))
        net_params_B = self.gen_paramsh_4(net_params_B)
        net_params_B = net_params_B.view(-1, self.num_param, self.num_function[0]) # (batch, 55, num_functions)

        #### generate boundary R
        boundary_R = torch.relu(self.fc_pd1(feature_resnet))
        boundary_R = torch.sigmoid(self.fc_pd2(boundary_R)) # (batch, num_functions)

        #### generate "polynomial coefficient" from symm-matrix B and boundary R
        net_params = []
        A_10x10 =[]
        for f in range(self.num_function[0]):
           coeff0, a0 = _pd_full(net_params_B[:,:,f].unsqueeze(-1), boundary_R[:,f].unsqueeze(-1))
           net_params.append(coeff0)
           A_10x10.append(a0)
        net_params = torch.cat(net_params, dim=-1) # (batch, 35, num_functions)
        A_10x10 = torch.cat(A_10x10, dim=-1) # (batch, 10, 10, num_functions)
        
        #### generate logits
        #logits = F.relu(self.fc1(net_params.view(-1, 35*self.num_functions)))
        #logits = self.fc2(logits)

        #### generate centers
        origins =  F.relu(self.orgh_1(feature_resnet))
        origins =  F.relu(self.orgh_2(origins))
        origins =  F.relu(self.orgh_3(origins))
        origins = torch.tanh(self.orgh_4(origins))  
        origins = torch.reshape(origins,(-1,3,self.num_function[0])) # (batch, 3, num_functions)

        #### geneate coefficients that reflects centers
        polycoeff_center = _gen_polycoeff_center(net_params, origins) # (batch, 35, num_functions)

        return polycoeff_center, origins, A_10x10




