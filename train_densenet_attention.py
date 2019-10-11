import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import joint_transforms
from config import duts_train_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir

from Densenet_attention import AADFNet
from torch.backends import cudnn

cudnn.benchmark = True

torch.manual_seed(2018)
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
# torch.cuda.set_device(0)

ckpt_path = './ckpt'


args = {
    'iter_num': 30000,
    'train_batch_size': 10,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': ''
}
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(400),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

train_set = ImageFolder(duts_train_path, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True, drop_last=True)

criterion = nn.BCEWithLogitsLoss().cuda()

save_points = range(8000, 30002, 1000)

def main():

    exp_name = 'AADFNet'
    train(exp_name)


def train(exp_name):

    net = AADFNet().cuda().train()
    net = nn.DataParallel(net, device_ids=[0, 1])

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])


    if len(args['snapshot']) > 0:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
    open(log_path, 'w').write(str(args) + '\n\n')
    print 'start to train'



    curr_iter = args['last_iter']
    while True:
        total_loss_record, loss1_record, loss2_record = AvgMeter(), AvgMeter(), AvgMeter()
        loss3_record, loss4_record = AvgMeter(), AvgMeter()
        loss2_2_record, loss3_2_record, loss4_2_record = AvgMeter(), AvgMeter(), AvgMeter()

        loss44_record, loss43_record, loss42_record, loss41_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss34_record, loss33_record, loss32_record, loss31_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss24_record, loss23_record, loss22_record, loss21_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss14_record, loss13_record, loss12_record, loss11_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()

            outputs4_2, outputs3_2, outputs2_2, outputs1, outputs2, outputs3, outputs4, \
                    predict41, predict42, predict43, predict44, \
                    predict31, predict32, predict33, predict34, \
                    predict21, predict22, predict23, predict24, \
                    predict11, predict12, predict13, predict14 = net(inputs)

            loss1 = criterion(outputs1, labels)
            loss2 = criterion(outputs2, labels)
            loss3 = criterion(outputs3, labels)
            loss4 = criterion(outputs4, labels)

            loss2_2 = criterion(outputs2_2, labels)
            loss3_2 = criterion(outputs3_2, labels)
            loss4_2 = criterion(outputs4_2, labels)

            loss44 = criterion(predict44, labels)
            loss43 = criterion(predict43, labels)
            loss42 = criterion(predict42, labels)
            loss41 = criterion(predict41, labels)

            loss34 = criterion(predict34, labels)
            loss33 = criterion(predict33, labels)
            loss32 = criterion(predict32, labels)
            loss31 = criterion(predict31, labels)

            loss24 = criterion(predict24, labels)
            loss23 = criterion(predict23, labels)
            loss22 = criterion(predict22, labels)
            loss21 = criterion(predict21, labels)

            loss14 = criterion(predict14, labels)
            loss13 = criterion(predict13, labels)
            loss12 = criterion(predict12, labels)
            loss11 = criterion(predict11, labels)

            total_loss = loss1 + loss2 + loss3 + loss4 + loss2_2 + loss3_2 + loss4_2 \
                         + (loss44 + loss43 + loss42 + loss41)/10 \
                         + (loss34 + loss33 + loss32 + loss31)/10 \
                         + (loss24 + loss23 + loss22 + loss21)/10 \
                         + (loss14 + loss13 + loss12 + loss11)/10

            total_loss = loss1 + loss2 + loss3 + loss4

            total_loss.backward()
            optimizer.step()

            total_loss_record.update(total_loss.item(), batch_size)
            loss1_record.update(loss1.item(), batch_size)
            loss2_record.update(loss2.item(), batch_size)
            loss3_record.update(loss3.item(), batch_size)
            loss4_record.update(loss4.item(), batch_size)

            loss2_2_record.update(loss2_2.item(), batch_size)
            loss3_2_record.update(loss3_2.item(), batch_size)
            loss4_2_record.update(loss4_2.item(), batch_size)

            loss44_record.update(loss44.item(), batch_size)
            loss43_record.update(loss43.item(), batch_size)
            loss42_record.update(loss42.item(), batch_size)
            loss41_record.update(loss41.item(), batch_size)

            loss34_record.update(loss34.item(), batch_size)
            loss33_record.update(loss33.item(), batch_size)
            loss32_record.update(loss32.item(), batch_size)
            loss31_record.update(loss31.item(), batch_size)

            loss24_record.update(loss24.item(), batch_size)
            loss23_record.update(loss23.item(), batch_size)
            loss22_record.update(loss22.item(), batch_size)
            loss21_record.update(loss21.item(), batch_size)

            loss14_record.update(loss14.item(), batch_size)
            loss13_record.update(loss13.item(), batch_size)
            loss12_record.update(loss12.item(), batch_size)
            loss11_record.update(loss11.item(), batch_size)


            curr_iter += 1

            log = '[iter %d], [total loss %.5f], ' \
                  '[loss4_2 %.5f], [loss3_2 %.5f], [loss2_2 %.5f], [loss1 %.5f], ' \
                  '[loss2 %.5f], [loss3 %.5f], [loss4 %.5f], ' \
                  '[loss44 %.5f], [loss43 %.5f], [loss42 %.5f], [loss41 %.5f], ' \
                  '[loss34 %.5f], [loss33 %.5f], [loss32 %.5f], [loss31 %.5f], ' \
                  '[loss24 %.5f], [loss23 %.5f], [loss22 %.5f], [loss21 %.5f], ' \
                  '[loss14 %.5f], [loss13 %.5f], [loss12 %.5f], [loss11 %.5f], ' \
                  '[lr %.13f]' % \
                  (curr_iter, total_loss_record.avg,
                   loss4_2_record.avg, loss3_2_record.avg,
                   loss2_2_record.avg, loss1_record.avg, loss2_record.avg,
                   loss3_record.avg, loss4_record.avg,
                   loss44_record.avg, loss43_record.avg, loss42_record.avg, loss41_record.avg,
                   loss34_record.avg, loss33_record.avg, loss32_record.avg, loss31_record.avg,
                   loss24_record.avg, loss23_record.avg, loss22_record.avg, loss21_record.avg,
                   loss14_record.avg, loss13_record.avg, loss12_record.avg, loss11_record.avg,
                   optimizer.param_groups[1]['lr'])

            print log
            open(log_path, 'a').write(log + '\n')


            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
                return


if __name__ == '__main__':
    main()
