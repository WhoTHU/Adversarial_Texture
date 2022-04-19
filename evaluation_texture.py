import os
import torch
import itertools
from tqdm import tqdm
import argparse
from scipy.interpolate import interp1d
from torchvision import transforms
unloader = transforms.ToPILImage()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fnmatch
import re

from yolo2 import load_data
from yolo2 import utils
from utils import *
from cfg import get_cfgs
from tps_grid_gen import TPSGridGen
from load_models import load_models
from generator_dim2 import GAN_dis

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--net', default='yolov2', help='target net name')
parser.add_argument('--method', default='TCEGA', help='method name')
parser.add_argument('--suffix', default=None, help='suffix name')
parser.add_argument('--gen_suffix', default=None, help='generator suffix name')
parser.add_argument('--device', default='cuda:0', help='')
parser.add_argument('--prepare_data', default=False, action='store_true', help='')
parser.add_argument('--epoch', type=int, default=None, help='')
parser.add_argument('--load_path', default=None, help='')
parser.add_argument('--load_path_z', default=None, help='')
parser.add_argument('--npz_dir', default=None, help='')
# parser.add_argument('--eval_times', type=int, default=1, help='evaluate multiple times')
pargs = parser.parse_args()


args, kwargs = get_cfgs(pargs.net, pargs.method, 'test')
if pargs.epoch is not None:
    args.n_epochs = pargs.epoch
if pargs.suffix is None:
    pargs.suffix = pargs.net + '_' + pargs.method

device = torch.device(pargs.device)

darknet_model = load_models(**kwargs)
darknet_model = darknet_model.eval().to(device)

class_names = utils.load_class_names('./data/coco.names')

target_func = lambda obj, cls: obj
patch_applier = load_data.PatchApplier().to(device)
patch_transformer = load_data.PatchTransformer().to(device)
if kwargs['name'] == 'ensemble':
    prob_extractor_yl2 = load_data.MaxProbExtractor(0, 80, target_func, 'yolov2').to(device)
    prob_extractor_yl3 = load_data.MaxProbExtractor(0, 80, target_func, 'yolov3').to(device)
else:
    prob_extractor = load_data.MaxProbExtractor(0, 80, target_func, kwargs['name']).to(device)
total_variation = load_data.TotalVariation().to(device)

target_control_points = torch.tensor(list(itertools.product(
    torch.arange(-1.0, 1.00001, 2.0 / 4),
    torch.arange(-1.0, 1.00001, 2.0 / 4),
)))

tps = TPSGridGen(torch.Size([300, 300]), target_control_points)
tps.to(device)

target_func = lambda obj, cls: obj
prob_extractor = load_data.MaxProbExtractor(0, 80, target_func, kwargs['name']).to(device)

results_dir = './results/result_' + pargs.suffix

if pargs.prepare_data:
    conf_thresh = 0.5
    nms_thresh = 0.4
    img_ori_dir = './data/INRIAPerson/Test/pos'
    img_dir = './data/test_padded'
    lab_dir = './data/test_lab_%s' % kwargs['name']
    data_nl = load_data.InriaDataset(img_ori_dir, None, kwargs['max_lab'], args.img_size, shuffle=False)
    loader_nl = torch.utils.data.DataLoader(data_nl, batch_size=args.batch_size, shuffle=False, num_workers=10)
    if lab_dir is not None:
        if not os.path.exists(lab_dir):
            os.makedirs(lab_dir)
    if img_dir is not None:
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
    print('preparing the test data')
    with torch.no_grad():
        for batch_idx, (data, labs) in tqdm(enumerate(loader_nl), total=len(loader_nl)):
            data = data.to(device)
            output = darknet_model(data)
            all_boxes = utils.get_region_boxes_general(output, darknet_model, conf_thresh, kwargs['name'])
            for i in range(data.size(0)):
                boxes = all_boxes[i]
                boxes = utils.nms(boxes, nms_thresh)
                boxes = torch.stack(boxes)
                new_boxes = boxes[:, [6, 0, 1, 2, 3]]
                new_boxes = new_boxes[new_boxes[:, 0] == 0]
                new_boxes = new_boxes.detach().cpu().numpy()
                if lab_dir is not None:
                    save_dir = os.path.join(lab_dir, labs[i])
                    np.savetxt(save_dir, new_boxes, fmt='%f')
                    img = unloader(data[i].detach().cpu())
                if img_dir is not None:
                    save_dir = os.path.join(img_dir, labs[i].replace('.txt', '.png'))
                    img.save(save_dir)
    print('preparing done')

img_dir_test = './data/test_padded'
lab_dir_test = './data/test_lab_%s' % kwargs['name']
test_data = load_data.InriaDataset(img_dir_test, lab_dir_test, kwargs['max_lab'], args.img_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=10)
loader = test_loader
epoch_length = len(loader)
print(f'One epoch is {len(loader)}')


def truths_length(truths):
    for i in range(50):
        if truths[i][1] == -1:
            return i


def label_filter(truths, labels=None):
    if labels is not None:
        new_truths = truths.new(truths.shape).fill_(-1)
        c = 0
        for t in truths:
            if t[0].item() in labels:
                new_truths[c] = t
                c = c + 1
        return new_truths


def test(model, loader, adv_patch=None, conf_thresh=0.5, nms_thresh=0.4, iou_thresh=0.5, num_of_samples=100,
         pooling=None, mode='ap'):
    model.eval()
    total = 0.0
    proposals = 0.0
    correct = 0.0
    batch_num = len(loader)

    with torch.no_grad():
        positives = []
        for batch_idx, (data, target) in tqdm(enumerate(loader), total=batch_num, position=0):
            data = data.to(device)

            if adv_patch is not None:
                target = target.to(device)
                adv_batch_t = patch_transformer(adv_patch, target, args.img_size, do_rotate=True, rand_loc=False,
                                                pooling=pooling)
                data = patch_applier(data, adv_batch_t)

            output = model(data)
            all_boxes = utils.get_region_boxes_general(output, model, conf_thresh, kwargs['name'])
            for i in range(len(all_boxes)):
                boxes = all_boxes[i]
                boxes = utils.nms(boxes, nms_thresh)
                truths = target[i].view(-1, 5)
                truths = label_filter(truths, labels=[0])
                truths = truths[:, 1:].tolist()
                num_gts = truths_length(truths)
                total = total + num_gts

                for j in range(len(boxes)):
                    if boxes[j][6].item() == 0:
                        best_iou = 0
                        best_index = 0

                        for box_gt in truths:
                            iou = utils.bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                            if iou > best_iou:
                                best_iou = iou
                                best_index = truths.index(box_gt)
                        if best_iou > iou_thresh:
                            del truths[best_index]
                            positives.append((boxes[j][4].item(), True))
                        else:
                            positives.append((boxes[j][4].item(), False))
        positives = sorted(positives, key=lambda d: d[0], reverse=True)

        tps = []
        fps = []
        confs = []
        tp_counter = 0
        fp_counter = 0
        for pos in positives:
            if pos[1]:
                tp_counter += 1
            else:
                fp_counter += 1
            tps.append(tp_counter)
            fps.append(fp_counter)
            confs.append(pos[0])

        precision = []
        recall = []
        for tp, fp in zip(tps, fps):
            recall.append(tp / total)
            precision.append(tp / (fp + tp))

    if len(precision) > 1 and len(recall) > 1:
        p = np.array(precision)
        r = np.array(recall)
        p_start = p[np.argmin(r)]
        samples = np.arange(0., 1., 1.0/num_of_samples)
        interpolated = interp1d(r, p, fill_value=(p_start, 0.), bounds_error=False)(samples)
        avg = sum(interpolated) / len(interpolated)
    elif len(precision) > 0 and len(recall) > 0:
        avg = precision[0] * recall[0]
    else:
        avg = float('nan')

    return precision, recall, avg, confs

if pargs.npz_dir is None:
    if pargs.method == 'RCA' or pargs.method == 'TCA':
        if pargs.load_path is None:
            result_dir = './results/result_' + pargs.net + '_' + pargs.method
            img_path = os.path.join(result_dir, 'patch%d.npy' % args.n_epochs)
        else:
            img_path = pargs.load_path
        cloth = torch.from_numpy(np.load(img_path)[:1])

    elif pargs.method == 'EGA' or pargs.method == 'TCEGA':
        gan = GAN_dis(DIM=128, z_dim=128, img_shape=(324, )*2)
        if pargs.load_path is None:
            result_dir = './results/result_' + pargs.net + '_' + pargs.method
            cpt = os.path.join(result_dir, pargs.net + '_' + pargs.method + '.pkl')
        else:
            cpt = pargs.load_path
        d = torch.load(cpt, map_location='cpu')
        gan.load_state_dict(d)
        gan.to(device)
        gan.eval()
        for p in gan.parameters():
            p.requires_grad = False
        if pargs.method == 'EGA':
            z_crop = torch.randn(1, 128, *args.z_size, device=device)
        else:
            # z load
            if pargs.load_path_z is None:
                result_dir = './results/result_' + pargs.net + '_' + pargs.method
                z_path = os.path.join(result_dir, 'z1000.npy')
            else:
                z_path = pargs.load_path_z
            z = np.load(z_path)
            z = torch.from_numpy(z).to(device)
            z_crop, _, _ = random_crop(z, args.z_crop_size, pos=args.z_pos, crop_type=args.z_crop_type)
        cloth = gan.generate(z_crop)
    else:
        raise ValueError

    adv_patch, x, y = random_crop(cloth, args.crop_size, pos=args.pos, crop_type=args.crop_type)
    adv_patch = adv_patch.to(device)
    adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])

    print(cloth.shape)
    print(adv_patch.shape)

    save_dir = './test_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, pargs.suffix)

    plt.figure(figsize=[15, 10])
    prec, rec, ap, confs = test(darknet_model, test_loader, adv_patch.detach().clone(), conf_thresh=0.01, pooling=args.pooling)
    np.savez(save_path, prec=prec, rec=rec, ap=ap, confs=confs, adv_patch=cloth.detach().cpu().numpy())
    print('AP is %.4f'% ap)
    plt.plot(rec, prec)
    leg = [pargs.suffix + ': ap %.3f' % ap]
    unloader(cloth[0]).save(save_path + '.png')
else:
    files = fnmatch.filter(os.listdir(pargs.npz_dir), '*.npz')
    order = {'RCA': 0, 'TCA': 1, 'EGA': 2, 'TCEGA': 3}
    files.sort()
    files.sort(key=lambda x: order[re.search('(RCA)|(TCA)|(EGA)|(TCEGA)', x).group()] if re.search('(RCA)|(TCA)|(EGA)|(TCEGA)', x) is not None else 1e5)

    leg = []
    for file in files:
        save_path = os.path.join(pargs.npz_dir, file)
        save_data = np.load(save_path, allow_pickle=True)
        save_data = save_data.values()
        prec, rec, ap, confs, clothi = list(save_data)
        plt.plot(rec, prec)
        leg.append(file.replace('.npz', '') + ', ap: %.3f' % ap)
        unloader(torch.from_numpy(clothi[0])).save(save_path.replace('.npz', '.png'))
    save_dir = pargs.npz_dir

plt.plot([0, 1], [0, 1], 'k--')
plt.legend(leg, loc=4)
plt.title('PR-curve')
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.ylim([0, 1.05])
plt.xlim([0, 1.05])
plt.savefig(os.path.join(save_dir, 'PR-curve.png'), dpi=300)





