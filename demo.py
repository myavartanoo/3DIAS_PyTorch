import os, sys
import argparse
from parse_config import ConfigParser

import torch
import model.model as module_arch
from utils.util import load_sample_images
from utils.generate_figure import generate_mesh

try:
    import open3d as o3d
except ImportError:
    print("### Open3D is NOT imported ###")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(config):
    # load model
    model = config.init_obj('arch', module_arch)
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    print("load model")

    # load image
    img, fname = load_sample_images(config.inputimg)
    img = torch.tensor(img).to(device)

    # run model
    polycoeff, origins, A_10x10 = model(img)

    # generate meshes
    meshlist, total_mesh, _ = generate_mesh(polycoeff, A_10x10)

    # generate figures
    output_dir = os.path.join("./output", fname)
    os.makedirs(output_dir, exist_ok=True)   
    if "open3d" in sys.modules:
        # save whole mesh
        o3d.io.write_triangle_mesh(os.path.join(output_dir, "total.ply"), total_mesh)    

        # save meshes of parts
        for i in range(len(meshlist)):
            o3d.io.write_triangle_mesh(os.path.join(output_dir, "parts_{}.ply".format(i)), meshlist[i])    
    
        # visualize with open3d
        if config.visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            for mesh in meshlist:
                vis.add_geometry(mesh)

            vis.run()
            vis.destroy_window()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='3DIAS_PyTorch')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    parser.add_argument('-t', '--tag', default=None, type=str,
                      help='experience name in tensorboard (default: None)')
    parser.add_argument('-i', '--inputimg', default=None, type=str,
                      help='input image file')
    parser.add_argument('--visualize', action='store_true', help='mesh visualizer with open3d')
                                      
    config = ConfigParser.from_args(parser.parse_args())

    main(config)