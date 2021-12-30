import torch
import numpy as np
import open3d as o3d
from torchmcubes import marching_cubes
from utils.util import gen_polynomial_orders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mesh_from_vf(verts, faces, valid_color):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(valid_color)
    return mesh

def generate_mesh(coeff, A_6x6):
    np.random.seed(222)
    color = np.random.rand(100, 3)

    polyorder = torch.from_numpy(gen_polynomial_orders(4)).to(device)
    valid_idx = [i for i in range(A_6x6.shape[3]) if torch.min(torch.eig(A_6x6[0,:,:,i])[0][:,0])<0]

    N = 128
    x, y, z = np.mgrid[:N, :N, :N]
    x = torch.from_numpy((x / (N-1)*2.2 -1.1).astype('float32')).cuda().unsqueeze(dim=3)#.reshape(-1,1)
    y = torch.from_numpy((y / (N-1)*2.2 -1.1).astype('float32')).cuda().unsqueeze(dim=3)#.reshape(-1,1)
    z = torch.from_numpy((z / (N-1)*2.2 -1.1).astype('float32')).cuda().unsqueeze(dim=3)#.reshape(-1,1)

    ### generate meshes 
    meshlist = []
    feasible_idx = []
    total_func_val = torch.ones(N, N, N).to(device)*100
    for i, idx in enumerate(valid_idx):
        func_val = (coeff[0,:,idx] * (x**polyorder[:,0] * y**polyorder[:,1] * z**polyorder[:,2])).sum(dim=3).detach()
        verts0, faces0 = marching_cubes(func_val, 0.0)
        if verts0.shape[0]>3: # if valid
            mesh0 = mesh_from_vf(verts0, faces0, color[valid_idx[i],:])
            meshlist.append(mesh0)
            feasible_idx.append(idx)

        # for whole shape
        total_func_val = torch.min(total_func_val, func_val) 

    ### generate a mesh for whole shape
    verts0, faces0 = marching_cubes(total_func_val, 0.0)
    total_mesh = mesh_from_vf(verts0, faces0, np.array([0.5, 0.5, 0.5])) # gray color

    return meshlist, total_mesh, feasible_idx

