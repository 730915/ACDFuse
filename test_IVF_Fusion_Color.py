# 导入自定义的网络模型组件
from net import Decoder,Encoder, INNModule
from ImprovementModule.APCA import APCA
# 导入HierarchicalFusionLayer
from ImprovementModule.HierarchicalFusionLayer import HierarchicalFusionLayer
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
import cv2  # 添加cv2用于彩色处理
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path=r"models/ACDFuse.pth"
for dataset_name in ["M3FD"]:
    print("\n"*2+"="*80)
    model_name="ACDFuse_Color"
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('test_img',dataset_name) 
    test_out_folder=os.path.join('ACDFuse',dataset_name)  # 修改输出文件夹名
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(Encoder()).to(device)
    Decoder = nn.DataParallel(Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(APCA(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(INNModule(num_layers=1)).to(device)
    HierarchicalFuser = nn.DataParallel(HierarchicalFusionLayer(dim=64, num_heads=4, num_layers=3, fusion_mode='sum')).to(device)

    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
    HierarchicalFuser.load_state_dict(torch.load(ckpt_path)['HierarchicalFuser'])
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()
    HierarchicalFuser.eval()

    # 确保输出文件夹存在
    os.makedirs(test_out_folder, exist_ok=True)
    
    with torch.no_grad():
        '''彩色融合版本''' # 结合test_color.py的彩色处理方法
        total_images = len(os.listdir(os.path.join(test_folder,"ir")))
        processed_count = 0
        skipped_count = 0
        
        for img_name in os.listdir(os.path.join(test_folder,"ir")):
            # 检查融合图像是否已存在
            output_filename = img_name.split(sep='.')[0] + ".png"
            output_path = os.path.join(test_out_folder, output_filename)
            
            if os.path.exists(output_path):
                print(f"跳过已融合的图像: {img_name}")
                skipped_count += 1
                continue
            
            print(f"正在融合图像: {img_name}")
            
            # 红外图像处理（灰度）
            data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            # 可见光图像处理（提取Y通道用于融合）
            data_VIS = cv2.split(image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='YCrCb'))[0][np.newaxis,np.newaxis, ...]/255.0
            
            # 获取可见光图像的色度信息（Cr, Cb通道）
            data_VIS_BGR = cv2.imread(os.path.join(test_folder,"vi",img_name))
            _, data_VIS_Cr, data_VIS_Cb = cv2.split(cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb))

            data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            # 使用改进的网络架构进行特征提取和融合
            feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_V_B,feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
            # 使用HierarchicalFusionLayer进一步融合基础和细节特征
            feature_F_Hierarchical = HierarchicalFuser(feature_F_B, feature_F_D)
            # 使用层次化融合特征生成融合图像
            data_Fuse, _ = Decoder(data_VIS, feature_F_Hierarchical, feature_F_Hierarchical)
            data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255.0).cpu().numpy())
            
            # 彩色重建：将融合的Y通道与原始Cr、Cb通道结合
            fi = fi.astype(np.uint8)
            ycrcb_fi = np.dstack((fi, data_VIS_Cr, data_VIS_Cb))
            rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
            img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder)
            processed_count += 1
        
        print(f"\n融合完成统计:")
        print(f"总图像数: {total_images}")
        print(f"新融合图像数: {processed_count}")
        print(f"跳过已存在图像数: {skipped_count}")


    eval_folder=test_out_folder  
    ori_img_folder=test_folder

