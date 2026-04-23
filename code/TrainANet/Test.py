import argparse
import csv
import os
import shutil

import cv2
import numpy as np
import torch
import tqdm

from lib.Network import Network
from utils.data_val import get_test_loader


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--ration', type=str, default='1', help='testing size')
parser.add_argument('--pth_path', type=str, default="../weight/ANet/")
parser.add_argument('--test_dataset_path', type=str, default="../data/LabelNoiseTrainDataset/")
parser.add_argument('--save_path', type=str, default="../pseudo_label/ANet/")
opt = parser.parse_args()


def compute_quality_score(mask_pred, edge_pred):
    mask_conf = np.mean(np.maximum(mask_pred, 1.0 - mask_pred))
    edge_conf = np.mean(np.clip(edge_pred, 0.0, 1.0))
    foreground_ratio = np.mean(mask_pred > 0.5)
    area_penalty = 1.0 - min(abs(foreground_ratio - 0.15) / 0.15, 1.0)
    score = 0.6 * mask_conf + 0.25 * edge_conf + 0.15 * max(area_penalty, 0.0)
    return float(np.clip(score, 0.0, 1.0))


opt.pth_path = opt.pth_path + opt.ration + '%/Net_epoch_best.pth'
datasets = ['CAMO_COD_generate_' + str(100 - int(opt.ration)) + '%']
with torch.no_grad():
    for _data_name in datasets:
        mae = []
        quality_rows = []
        data_path = opt.test_dataset_path + '/{}/'.format(_data_name)
        save_path = opt.save_path + '/' + _data_name
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path + '/mask', exist_ok=True)
        os.makedirs(save_path + '/edge', exist_ok=True)

        model = Network()
        weights = torch.load(opt.pth_path)
        model.load_state_dict(weights)

        model.cuda()
        model.eval()

        image_root = '{}/image/'.format(data_path)
        gt_root = '{}/mask/'.format(data_path)
        test_loader = get_test_loader(image_root, gt_root, 12, opt.testsize)

        for i, (image, bbox_image, gt, [H, W], name) in tqdm.tqdm(enumerate(test_loader, start=1)):
            gt = gt.cuda()
            bbox_image = bbox_image.cuda()
            image = image.cuda()
            result = model(image, bbox_image)
            res = result[4]
            edge = result[8]
            res = res.sigmoid()
            mae.append(torch.mean(torch.abs(gt - res)).data.cpu().numpy())
            edge = edge.squeeze().detach().cpu().numpy()
            res = res.squeeze().detach().cpu().numpy()

            for j in range(len(res)):
                pre = cv2.resize(res[j], dsize=(H[j].item(), W[j].item()))
                ed = cv2.resize(edge[j], dsize=(H[j].item(), W[j].item()))
                score = compute_quality_score(pre, ed)
                quality_rows.append({'name': name[j].replace(".jpg", ".png"), 'score': f'{score:.6f}'})
                cv2.imwrite(save_path + '/mask/' + name[j].replace(".jpg", ".png"), pre * 255.)
                cv2.imwrite(save_path + '/edge/' + name[j].replace(".jpg", ".png"), ed * 255.)

        with open(os.path.join(save_path, 'quality_scores.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'score'])
            writer.writeheader()
            writer.writerows(quality_rows)
        print(np.mean(mae))

src_image_dir = "../data/LabelNoiseTrainDataset/CAMO_COD_generate_" + str(100 - int(opt.ration)) + "%/image"
dst_image_dir = "../pseudo_label/ANet/CAMO_COD_generate_" + str(100 - int(opt.ration)) + "%/image"
if os.path.exists(dst_image_dir):
    shutil.rmtree(dst_image_dir)
shutil.copytree(src_image_dir, dst_image_dir)
