import torch.nn as nn
import torch.nn.init as init
from utils.general_utils import  batch_quaternion_multiply
from scene.hexplane import HexPlaneField
import torch
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.args = args
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)

    def create_net(self):
        grid_out_dim = self.grid.feat_dim
        self.feature_out = [nn.Linear(grid_out_dim,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))

    def query_time(self, rays_pts_emb,time_emb):
        # import torch
        # print(torch.isnan(rays_pts_emb).sum())
        grid_feature = self.grid(rays_pts_emb, time_emb[:,:1])
        hidden = self.feature_out(grid_feature)   
        return hidden

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity_emb = None,shs_emb=None, time_emb=None):
        hidden = self.query_time(rays_pts_emb, time_emb)

        if self.args.no_dx:
            pts = rays_pts_emb
        else:
            dx = self.pos_deform(hidden)
            pts = rays_pts_emb + dx
        if self.args.no_ds:
            
            scales = scales_emb
        else:
            ds = self.scales_deform(hidden)
            scales = scales_emb + ds
            
        if self.args.no_dr:
            rotations = rotations_emb
        else:
            dr = self.rotations_deform(hidden)
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb + dr

        if self.args.no_do:
            opacity = opacity_emb
        else:
            do = self.opacity_deform(hidden) 
            opacity = opacity_emb + do

        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])
            shs = shs_emb+ dshs

        return pts, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
    
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        defor_depth= args.defor_depth
        self.deformation_net = Deformation(W=net_width, D=defor_depth, args=args)
        self.apply(initialize_weights)

    @property
    def get_aabb(self):
        return self.deformation_net.get_aabb
    
    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        means3D, scales, rotations, opacity, shs = self.deformation_net( point,
                                                  scales,
                                                rotations,
                                                opacity,
                                                shs,
                                                times_sel)
        return means3D, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters()
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            # init.constant_(m.bias, 0) #AI建议
            init.xavier_uniform_(m.weight,gain=1)
