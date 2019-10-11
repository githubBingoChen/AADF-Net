import numpy as np
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torch import nn
import torch.nn.functional as F

from config import ecssd_path, hkuis_path, pascals_path, sod_path, dutomron_path, duts_path, thur15k_path, soc_path
from misc import check_mkdir, crf_refine, AvgMeter, cal_precision_recall_mae, cal_fmeasure, cal_fmeasure_both

from Densenet_attention import AADFNet

torch.manual_seed(2018)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# set which gpu to use
# torch.cuda.set_device(1)

# the following two args specify the location of the file of trained model (pth extension)
# you should have the pth file in the folder './$ckpt_path$/$exp_name$'
args = {
    'snapshot': '30000',  # your snapshot filename (exclude extension name)
    'crf_refine': True,  # whether to use crf to refine results
    'save_results': False  # whether to save the resulting masks
}

ckpt_path = './ckpt'
exp_name = 'AADFNet'
exp_predict = args['snapshot'] + ' predict1'




img_transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()
to_test = {'ecssd': ecssd_path, 'hkuis': hkuis_path, 'pascal': pascals_path, 'dutomron': dutomron_path, 'duts': duts_path, 'sod': sod_path,'soc': soc_path}

def main():
    net = AADFNet().cuda()
    net = nn.DataParallel(net, device_ids=[0])

    print exp_name + 'crf: '+ str(args['crf_refine'])
    print 'load snapshot \'%s\' for testing' % args['snapshot']
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
    net.eval()

    with torch.no_grad():
        results = {}

        for name, root in to_test.iteritems():

            precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
            mae_record = AvgMeter()
            time_record = AvgMeter()

            img_list = [os.path.splitext(f)[0] for f in os.listdir(root) if f.endswith('.jpg')]

            for idx, img_name in enumerate(img_list):
                img_name = img_list[idx]
                print 'predicting for %s: %d / %d' % (name, idx + 1, len(img_list))
                check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

                start = time.time()
                img = Image.open(os.path.join(root, img_name + '.jpg')).convert('RGB')
                img_var = Variable(img_transform(img).unsqueeze(0), volatile=True).cuda()
                prediction = net(img_var)
                W, H = img.size
                prediction = F.upsample_bilinear(prediction, size=(H, W))
                prediction = np.array(to_pil(prediction.data.squeeze(0).cpu()))


                if args['crf_refine']:
                    prediction = crf_refine(np.array(img), prediction)

                end = time.time()

                gt = np.array(Image.open(os.path.join(root, img_name + '.png')).convert('L'))
                precision, recall, mae = cal_precision_recall_mae(prediction, gt)
                for pidx, pdata in enumerate(zip(precision, recall)):
                    p, r = pdata
                    precision_record[pidx].update(p)
                    recall_record[pidx].update(r)

                mae_record.update(mae)
                time_record.update(end-start)



                if args['save_results']:
                    Image.fromarray(prediction).save(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (
                        exp_name, name, args['snapshot']), img_name + '.png'))

            max_fmeasure, mean_fmeasure = cal_fmeasure_both([precord.avg for precord in precision_record],
                                                            [rrecord.avg for rrecord in recall_record])
            results[name] = {'max_fmeasure': max_fmeasure, 'mae': mae_record.avg, 'mean_fmeasure': mean_fmeasure}

        print 'test results:'
        print results

        with open('Result', 'a') as f:
            if args['crf_refine']:
                f.write('with CRF')

            f.write('Runing time %.6f \n' % time_record.avg)
            f.write('\n%s\n  iter:%s\n' % (exp_name, args['snapshot']))
            for name, value in results.iteritems():
                f.write('%s: mean_fmeasure: %.10f, mae: %.10f, max_fmeasure: %.10f\n' % (
                    name, value['mean_fmeasure'], value['mae'], value['max_fmeasure']))





if __name__ == '__main__':
    main()
