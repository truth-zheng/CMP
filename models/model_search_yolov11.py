from torch.nn import init
import torch
import torch.nn as nn
from torchvision.ops import roi_align
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.cmp import CMP
from models.pose import Block, ConvExpandReduce
from models.gcn import Block2
from nets.nn import YOLODetector

class Search(CMP):
    def __init__(self, config):
        super().__init__(config,)

        self.be_hard = config.get('be_hard', False)
        self.be_pose_img = config.get('be_pose_img', False)
        self.be_pose_conv = config.get('pose_conv', False)
        self.use_yolo_gcn = config.get('use_yolo_gcn', False)

        # 默认参数
        self.yolo_person_class_id = config.get('yolo_person_class_id', 0)   # COCO 里 person 是 0
        self.yolo_conf_threshold = config.get('yolo_conf_threshold', 0.2)

        if self.be_pose_img:
            self.pose_block = Block()
            self.init_params.extend(['pose_block.' + n for n, _ in self.pose_block.named_parameters()])
            if self.be_pose_conv:
                print('pose_conv')
                self.pose_conv = ConvExpandReduce()
                self.init_params.extend(['pose_conv.' + n for n, _ in self.pose_conv.named_parameters()])

        if self.use_yolo_gcn:
            # 初始化YOLO检测器
            self.yolo_detector = YOLODetector()

            # ---------- 加载本地权重 ----------
            yolo_weight_path = config.get('yolo_weight_path')
            if os.path.exists(yolo_weight_path):
                yolo_model = torch.load(yolo_weight_path, map_location="cpu")['model'].float()
                state_dict = yolo_model.state_dict()
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = "model." + k  # 给每个 key 加上 'model.' 前缀
                    new_state_dict[new_key] = v
                self.yolo_detector.load_state_dict(new_state_dict, strict=True)
                print(f"[YOLO] Loaded weights from {yolo_weight_path}")
            else:
                print(f"[YOLO] Warning: weight file {yolo_weight_path} not found. Using random init.")

            # ---------- 冻结参数 ----------
            for param in self.yolo_detector.parameters():
                param.requires_grad = False

            # ---------- 固定 BN / Dropout ----------
            self.yolo_detector.eval()  # 保证BN/Dropout不受外部train()影响
            for m in self.yolo_detector.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()  # 固定BN层

                
            # 初始化GCN模块
            self.gcn = GCNBlock() #config['gcn_config']
            self.init_params.extend(['gcn.' + n for n, _ in self.gcn.named_parameters()])
            
            # 初始化用于融合人物特征的交叉注意力模块
            self.person_block = Block2()
            self.init_params.extend(['person_block.' + n for n, _ in self.person_block.named_parameters()])

    def detect_persons_and_extract_features(self, image, save_dir="debug_vis"):
        """
        使用YOLO检测人物并提取ROI特征，支持批量处理，并对齐人数
        增加可视化保存，保存检测框 + ROI特征图
        """
        os.makedirs(save_dir, exist_ok=True)

        batch_size = image.size(0)
        self.yolo_detector.eval()
        # 使用YOLO检测人物
        with torch.no_grad():
            detections, yolo_feature_maps = self.yolo_detector(image)
            boxes, scores, classes = detections
        
        feature_map = yolo_feature_maps[2]  # 选择中间尺度的特征图 14 14
        spatial_scale = feature_map.shape[-1] / image.shape[-1]  # 修正scale
        # print(f"Feature map shape: {feature_map.shape}, spatial_scale: {spatial_scale}") Feature map shape: torch.Size([22, 384, 28, 28]), spatial_scale: 0.125
        # os.exit(1)
        
        all_person_features = []
        person_counts = []
        
        for i in range(batch_size):
            img_boxes = boxes[i]
            img_scores = scores[i]
            img_classes = classes[i]
            
            # 过滤出人物
            person_mask = img_classes == self.yolo_person_class_id
            person_boxes = img_boxes[person_mask]
            person_scores = img_scores[person_mask]
            
            # 过滤低置信度
            high_conf_mask = person_scores > self.yolo_conf_threshold
            person_boxes = person_boxes[high_conf_mask]
            
            if len(person_boxes) == 0:
                all_person_features.append(None)
                person_counts.append(0)
                continue
            
            # 转换为像素坐标
            height, width = image.shape[2], image.shape[3]
            pixel_boxes = person_boxes.clone()
            pixel_boxes[:, 0] *= width
            pixel_boxes[:, 1] *= height
            pixel_boxes[:, 2] *= width
            pixel_boxes[:, 3] *= height
            
            # 生成 rois
            rois = torch.cat([
                torch.full((pixel_boxes.size(0), 1), i, device=image.device, dtype=torch.float32),
                pixel_boxes
            ], dim=1)
            
            roi_features = roi_align(
                feature_map, 
                rois, 
                output_size=(7, 7),
                spatial_scale=spatial_scale,
                sampling_ratio=2
            )
            
            all_person_features.append(roi_features)
            person_counts.append(roi_features.size(0))

            # ========= 可视化保存 =========
            
            # 先把tensor转换为float32，再转numpy
# 保存每个ROI的原图裁剪结果

            # ====== 可视化特征图上的 ROI 裁切 ======
            # feat_map = feature_map[i].detach().cpu()  # [C, Hf, Wf]\
            # print(f"Feature map shape: {feat_map.shape}")
            # feat_map_vis = feat_map.mean(0).to(torch.float32).numpy()   # [Hf, Wf]，取均值作为可视化

            # # 归一化到0-255
            # feat_map_vis = (feat_map_vis - feat_map_vis.min()) / (feat_map_vis.max() - feat_map_vis.min() + 1e-5)
            # feat_map_vis = (feat_map_vis * 255).astype(np.uint8)
            # feat_map_vis = cv2.applyColorMap(feat_map_vis, cv2.COLORMAP_VIRIDIS)

            # # 将 box 映射到特征图尺度
            # scale_x = feat_map.shape[2] / width
            # scale_y = feat_map.shape[1] / height
            # for j, box in enumerate(pixel_boxes.to(torch.float32).cpu().numpy()):
            #     x1, y1, x2, y2 = map(int, box)
            #     fx1, fy1, fx2, fy2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
            #     cv2.rectangle(feat_map_vis, (fx1, fy1), (fx2, fy2), (0, 0, 255), 1)

            # # 保存带框的特征图
            # cv2.imwrite(os.path.join(save_dir, f"batch{i}_featuremap_det.jpg"), feat_map_vis)

            # # # 保存每个ROI裁切的7x7特征
            # # for j in range(roi_features.shape[0]):
            # #     roi_patch = roi_features[j].mean(0).to(torch.float32).cpu().numpy()  # [7,7]
            # #     plt.imshow(roi_patch, cmap="viridis")
            # #     plt.savefig(os.path.join(save_dir, f"batch{i}_roi_feat{j}.png"))
            # #     plt.close()

            # img_np = image[i].permute(1, 2, 0).detach().cpu().to(torch.float32).numpy()  # [H,W,C], float32
            # img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)  # 转 uint8 并防止越界
            # img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            # pixel_boxes = pixel_boxes.to(torch.float32)
            # roi_features = roi_features.to(torch.float32)


            # for j, box in enumerate(pixel_boxes.to(torch.float32).cpu().numpy()):
            #     x1, y1, x2, y2 = map(int, box)
            #     roi_crop = img_np[y1:y2, x1:x2]  # 直接在原图上裁切
            #     cv2.imwrite(os.path.join(save_dir, f"batch{i}_roi{j}.jpg"), roi_crop)



            # # 保存原图
            # cv2.imwrite(os.path.join(save_dir, f"batch{i}_orig.jpg"), img_np)
            # # 绘制检测框
            # for box in pixel_boxes.cpu().numpy():
            #     x1, y1, x2, y2 = map(int, box)
            #     cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # # 保存图片
            # cv2.imwrite(os.path.join(save_dir, f"batch{i}_det.jpg"), img_np)

            # # # 保存ROI特征 (numpy)
            # # np.save(os.path.join(save_dir, f"batch{i}_roi.npy"), roi_features.detach().cpu().numpy())

            # # # 可选：保存第一张ROI的可视化
            # # roi_vis = roi_features[0].mean(0).cpu().numpy()  # C,h,w → h,w
            # # plt.imshow(roi_vis, cmap="viridis")
            # # plt.colorbar()
            # # plt.savefig(os.path.join(save_dir, f"batch{i}_roi0.png"))
            # # plt.close()

        
        # 找出 batch 内最大人数
        if any(f is not None for f in all_person_features):
            C = next(f.size(1) for f in all_person_features if f is not None)
        else:
            C = 512  # fallback
        
        max_num_persons = max(person_counts) if person_counts else 0
        padded_features = []
        masks = []
        for f in all_person_features:
            if f is None:
                f = torch.zeros(0, C, 7, 7, device=image.device)
            pad_size = max_num_persons - f.size(0)
            if pad_size > 0:
                pad = torch.zeros(pad_size, C, 7, 7, device=image.device)
                f = torch.cat([f, pad], dim=0)
            padded_features.append(f)
            
            mask = torch.zeros(max_num_persons, device=image.device)
            mask[:f.size(0) - pad_size] = 1
            masks.append(mask)
        
        if max_num_persons > 0:
            person_features = torch.stack(padded_features, dim=0)
            masks = torch.stack(masks, dim=0)
        else:
            person_features, masks = None, None
        
        return person_features, masks


    def forward(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                idx=None, text_ids_eda=None, text_atts_eda=None,
                pose=None, hard_i=None, hard_i_pose=None, hard_text_ids=None, hard_text_atts=None,
                ):
        # import torchvision.utils
        # print(image.shape)
        # for i in range(image.size(0)):
        #     img = image[i].detach().cpu()
        #     # 如果是归一化到[0,1]的float，可以直接保存
        #     torchvision.utils.save_image(img, f"debug_vis/forward_batch{i}.jpg")
        # os._exit(0)
        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)

        if self.be_pose_img:
            if self.be_pose_conv:
                pose = self.pose_conv(pose)
            pose, _ = self.get_vision_embeds(pose)
            image_embeds = self.pose_block(image_embeds, pose)

        if self.use_yolo_gcn:
            # 检测人物并提取ROI特征
            person_features, masks = self.detect_persons_and_extract_features(image)
            # print(f"Person features shape: {person_features.shape if person_features is not None else None}")
            # print(person_features)
            # if person_features is not None:
            #     if torch.isnan(person_features).any():
            #         print("person_features 包含 NaN")
            #     elif person_features.abs().sum() > 0:
            #         print("person_features 包含非零元素")
            #     else:
            #         print("person_features 全为零")
            # else:
            #     print("person_features 为 None")
            if person_features is not None:
                person_features = self.gcn(person_features, mask=masks)
                # print(image_embeds.shape, person_features.shape) torch.Size([22, 50, 1024]) torch.Size([22, 1, 1024])
                image_embeds = self.person_block(image_embeds, person_features)
                # import torch.nn.functional as F
                # image_embeds = F.normalize(image_embeds, dim=-1)
                # print(person_features)
                # print(image_embeds)
                # print(f"Image embeds after GCN shape: {image_embeds.shape}") # Image embeds after GCN shape: torch.Size([22, 50, 1024])
            else:
                image_embeds = image_embeds

        # print(f"Image embeds shape: {image_embeds.shape}, Text embeds shape: {text_embeds.shape}")
        image_feat, text_feat = self.get_image_feat(image_embeds), self.get_text_feat(text_embeds)
        loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                          text_embeds, text_atts, text_feat, idx=idx)

        # eda
        text_embeds_eda = self.get_text_embeds(text_ids_eda, text_atts_eda)
        text_feat_eda = self.get_text_feat(text_embeds_eda)
        loss_itc_eda = self.get_contrastive_loss(image_feat, text_feat_eda, idx=idx)
        loss_itm_eda = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                              text_embeds_eda, text_atts_eda, text_feat_eda, idx=idx, )
        loss_itc = loss_itc + 0.8 * loss_itc_eda
        loss_itm = loss_itm + 0.8 * loss_itm_eda

        loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts,
                                     masked_pos, masked_ids, )

        if self.be_hard:
            image_embeds_hard, image_atts_hard = self.get_vision_embeds(hard_i)
            text_embeds_hard = self.get_text_embeds(hard_text_ids, hard_text_atts)

            if self.be_pose_img:
                if self.be_pose_conv:
                    hard_i_pose = self.pose_conv(hard_i_pose)

                hard_pose, _ = self.get_vision_embeds(hard_i_pose)
                image_embeds_hard = self.pose_block(image_embeds_hard, hard_pose)

            loss_itm_hard = self.get_matching_loss_hard(image_embeds, image_atts, image_embeds_hard, image_atts_hard,
                                                        text_embeds, text_atts, text_embeds_hard, hard_text_atts)
            loss_itm = loss_itm + loss_itm_hard

        return loss_itc, loss_itm, loss_mlm


class GCNBlock(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.input_dim = 768
        self.hidden_dim =  512
        self.output_dim = 1024
        
        # 图卷积层
        self.gcn_layers = nn.ModuleList([
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_dim)
        ])

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            self.input_dim, 
            num_heads=4, 
            batch_first=True,
            dropout=0.1  # 添加dropout防止过拟合
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，防止梯度爆炸"""
        for layer in self.gcn_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        
    def forward(self, x, mask=None):
        # 检查输入是否为NaN
        if torch.isnan(x).any():
            print("⚠️ GCN输入包含NaN!")
            # 将NaN替换为0
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        batch_size, num_persons, channels, h, w = x.shape
        
        # 全局平均池化
        x = x.view(batch_size * num_persons, channels, h, w)
        x = torch.mean(x, dim=(2, 3))  # [batch_size*num_persons, C]
        x = x.view(batch_size, num_persons, channels)
        
        # 检查是否全为零
        if x.abs().sum() == 0:
            print("⚠️ 人物特征全为零，跳过GCN处理")
            # 返回安全的零向量
            return torch.zeros(batch_size, 1, self.output_dim, device=x.device)
        
        # 处理mask
        if mask is not None:
            # 确保mask是bool类型
            mask = mask.bool()
            # 检查是否有有效的人物
            valid_batches = mask.any(dim=1)
            
            # 对于没有有效人物的batch，直接返回零向量
            if not valid_batches.all():
                # print(f"⚠️ 部分batch没有有效人物: {valid_batches}")
                output = torch.zeros(batch_size, 1, self.output_dim, device=x.device)
                # 只对有效batch进行处理
                valid_indices = valid_batches.nonzero(as_tuple=True)[0]
                if len(valid_indices) > 0:
                    valid_x = x[valid_indices]
                    valid_mask = mask[valid_indices]
                    
                    # 应用mask
                    valid_x = valid_x * valid_mask.unsqueeze(-1)
                    
                    # 注意力机制
                    attn_output, _ = self.attention(
                        valid_x, valid_x, valid_x, 
                        key_padding_mask=~valid_mask  # 注意这里取反
                    )
                    valid_x = valid_x + attn_output
                    
                    # GCN层
                    for i, layer in enumerate(self.gcn_layers):
                        valid_x = layer(valid_x)
                        if i < len(self.gcn_layers) - 1:
                            valid_x = self.relu(valid_x)
                            valid_x = self.dropout(valid_x)
                    
                    # 加权平均
                    valid_output = (valid_x * valid_mask.unsqueeze(-1)).sum(dim=1) 
                    valid_output = valid_output / valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
                    valid_output = valid_output.unsqueeze(1)
                    
                    output[valid_indices] = valid_output
                
                return output
            
            # 所有batch都有有效人物
            x = x * mask.unsqueeze(-1)
        
        # 正常的注意力计算
        key_padding_mask = ~mask if mask is not None else None
        attn_output, _ = self.attention(
            x, x, x, 
            key_padding_mask=key_padding_mask
        )
        x = x + attn_output
        
        # 检查中间结果
        if torch.isnan(x).any():
            print("⚠️ 注意力后出现NaN!")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        # GCN层
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x)
            if i < len(self.gcn_layers) - 1:
                x = self.relu(x)
                x = self.dropout(x)
            
            # 检查每层输出
            if torch.isnan(x).any():
                print(f"⚠️ 第{i}层GCN后出现NaN!")
                x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        # 最终聚合
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(dim=1) 
            x = x / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            x = torch.mean(x, dim=1)
        
        x = x.unsqueeze(1)  # [B, 1, C_out]
        
        # 最终检查
        if torch.isnan(x).any():
            print("⚠️ 最终输出包含NaN，使用零替换!")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        return x
    
class GCNBlock_0(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = 768
        self.hidden_dim =  512
        self.output_dim = 1024
        
        # 图卷积层
        self.gcn_layers = nn.ModuleList([
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_dim)
        ])

        # 注意力机制，用于计算人物之间的关系
        self.attention = nn.MultiheadAttention(self.input_dim, num_heads=4, batch_first=True)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        # x: [batch_size, num_persons, C, h, w]
        batch_size, num_persons, channels, h, w = x.shape
        
        x = x.view(batch_size * num_persons, channels, h, w)
        x = torch.mean(x, dim=(2, 3))  # [batch_size*num_persons, C]
        x = x.view(batch_size, num_persons, channels)
        
        if mask is not None:
            # 把padding位置置零
            x = x * mask.unsqueeze(-1)
            if mask.sum() == 0:
                mask = torch.ones_like(mask)

        if x is not None:
            if torch.isnan(x).any():
                print("person_features 包含 NaN")
            elif x.abs().sum() > 0:
                print("person_features 包含非零元素")
            else:
                print("person_features 全为零")
        os.eixt()
        attn_output, _ = self.attention(x, x, x, key_padding_mask=(mask == 0) if mask is not None else None)
        x = x + attn_output

        if x is not None:
            if torch.isnan(x).any():
                print("person_features 包含 NaN")
            elif x.abs().sum() > 0:
                print("person_features 包含非零元素")
            else:
                print("person_features 全为零")
        os.eixt()

        for i, layer in enumerate(self.gcn_layers):
            x = layer(x)
            if i < len(self.gcn_layers) - 1:
                x = self.relu(x)
                x = self.dropout(x)


        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            x = torch.mean(x, dim=1)

        x = x.unsqueeze(1)  # [B, 1, C_out]
        # print(f"GCN output shape: {x.shape}")
        return x
    


class GCNBlock6(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, 1024),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        """
        x: [B, N, C]
        mask: [B, N]  1 表示有效 token, 0 表示 padding
        """
        # -------- 安全处理 mask --------
        if mask is not None:
            # 如果整个 batch 的某个样本全是 0，就强制留一个 token 有效，避免全屏蔽
            safe_mask = mask.clone()
            for b in range(safe_mask.size(0)):
                if safe_mask[b].sum() == 0:
                    safe_mask[b, 0] = 1
            key_padding_mask = (safe_mask == 0)  # True = 要屏蔽
        else:
            key_padding_mask = None

        # -------- Multi-head Attention --------
        residual = x
        attn_output, _ = self.attention(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=key_padding_mask
        )

        # 防止 NaN
        # attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=0.0, neginf=0.0)
        x = residual + attn_output

        # -------- Feed Forward --------
        residual = x
        x = residual + self.mlp(self.norm2(x))

        # # 再做一次 NaN 防护
        # x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

class GCNBlock1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.get('input_dim', 512)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.output_dim = config.get('output_dim', 512)
        
        # 图卷积层
        self.gcn_layers = nn.ModuleList([
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_dim)
        ])
        
        # 注意力机制，用于计算人物之间的关系
        self.attention = nn.MultiheadAttention(self.input_dim, num_heads=4, batch_first=True)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.person_feat_proj = nn.Linear(768, 1024)
        
    def forward(self, x, mask=None):
        # x: [batch_size, num_persons, C, h, w]
        batch_size, num_persons, channels, h, w = x.shape
        
        x = x.view(batch_size * num_persons, channels, h, w)
        x = torch.mean(x, dim=(2, 3))  # [batch_size*num_persons, C]
        x = x.view(batch_size, num_persons, channels)
        
        if mask is not None:
            # 把padding位置置零
            x = x * mask.unsqueeze(-1)
        
        attn_output, _ = self.attention(x, x, x, key_padding_mask=(mask == 0) if mask is not None else None)
        x = x + attn_output
        
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x)
            if i < len(self.gcn_layers) - 1:
                x = self.relu(x)
                x = self.dropout(x)
        
        if mask is not None:
            x = x * mask.unsqueeze(-1)
            
        attn_output, _ = self.attention(x, x, x, key_padding_mask=(mask == 0) if mask is not None else None)
        x = x + attn_output

        for i, layer in enumerate(self.gcn_layers):
            x = layer(x)
            if i < len(self.gcn_layers) - 1:
                x = self.relu(x)
                x = self.dropout(x)
        x = self.person_feat_proj(x)
        print(f"GCN output shape: {x.shape}")

        # 不再做 sum/mean pooling，直接返回 [B, N, C_out]
        # x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        # else:
        #     x = torch.mean(x, dim=1)
        return x  # [B, N, C_out]
