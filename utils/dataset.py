import torch.utils.data as Data
import h5py
import numpy as np
import torch

class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['ir_patchs'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        IR = np.array(h5f['ir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])
        h5f.close()
        return torch.Tensor(VIS), torch.Tensor(IR)


class H5DatasetDetection(Data.Dataset):
    """
    支持检测标注的 H5 数据集

    数据结构 (H5 文件):
        ir_patchs/
            key_0: [H, W] infrared patch
            key_1: [H, W] infrared patch
            ...
        vis_patchs/
            key_0: [H, W] visible patch
            key_1: [H, W] visible patch
            ...
        annotations/
            key_0: {
                'boxes': [[x1, y1, x2, y2], ...],  # normalized to [0,1]
                'labels': [class_id, ...],  # 0-indexed
            }
            ...
    """
    def __init__(self, h5file_path, img_size=640):
        self.h5file_path = h5file_path
        self.img_size = img_size
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['ir_patchs'].keys())
        self.has_annotations = 'annotations' in h5f
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]

        IR = np.array(h5f['ir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])

        # 转换为 [C, H, W] 格式
        if IR.ndim == 2:
            IR = IR[np.newaxis, ...]
        if VIS.ndim == 2:
            VIS = VIS[np.newaxis, ...]

        vis = torch.Tensor(VIS)
        ir = torch.Tensor(IR)

        if self.has_annotations:
            ann = h5f['annotations'][key]
            boxes = np.array(ann['boxes'])  # shape: [N, 4], normalized to [0,1]
            labels = np.array(ann['labels'])  # shape: [N]
            h5f.close()

            # 转换为 torch 格式: [N, 4] (x1, y1, x2, y2), [N]
            boxes = torch.Tensor(boxes)
            labels = torch.LongTensor(labels)

            return vis, ir, {'boxes': boxes, 'labels': labels}
        else:
            h5f.close()
            return vis, ir, None


class COCO DetectionDataset(Data.Dataset):
    """
    COCO 格式检测数据集适配器

    适用于 FLIR、LLVIP、M3FD 等公开数据集

    数据格式:
        images/: {id: img.npy}
        annotations/: {image_id: {'boxes': [[x,y,w,h],...], 'labels': [...]}}
    """
    def __init__(self, data_dir, img_size=640):
        self.data_dir = data_dir
        self.img_size = img_size
        self.img_ids = []
        # TODO: 实现数据加载逻辑
        raise NotImplementedError("COCO 格式数据集适配器待实现")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        raise NotImplementedError("COCO 格式数据集适配器待实现")