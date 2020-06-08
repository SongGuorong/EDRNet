# coding=utf-8
import torch
import torch.nn as nn
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model.EDRNet import EDRNet
import pytorch_ssim
import pytorch_iou
from torch.backends import cudnn
import utils.func as func


cudnn.benchmark = True
torch.manual_seed(2018)
torch.cuda.manual_seed_all(2018)

# load training default parameters
args = {
    'epoch': 610,
    'batch_size': 8,
    'lr': 0.001,
    'workers': 5,
    'tra_img_dir': './Data/trainingDataset/imgs_train/',              # path of training images
    'tra_lbl_dir': './Data/trainingDataset/masks_train/',             # path of training labels
    'image_ext': '.bmp',
    'label_ext': '.png',
    'checkpoint': './trained_models/',
}

chkpt_dir = args['checkpoint']
func.check_mkdir(chkpt_dir)

# ------- 1. define loss function --------
bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


def train_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + ssim_out + iou_out

    return loss


def muti_loss_fusion(s_out, s0, s1, s2, s3, s4, sb, labels_v):
    loss0 = train_loss(s_out, labels_v)
    loss1 = train_loss(s0, labels_v)
    loss2 = train_loss(s1, labels_v)
    loss3 = train_loss(s2, labels_v)
    loss4 = train_loss(s3, labels_v)
    loss5 = train_loss(s4, labels_v)
    loss6 = train_loss(sb, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss


def main():
    print(args)
    tra_img_name_list = glob.glob(args['tra_img_dir'] + '*' + args['image_ext'])
    tra_lbl_name_list = []

    for img_path in tra_img_name_list:
        img_name = img_path.split("/")[-1]          # ubuntu
        imgIdx = img_name.split(".")[0]
        tra_lbl_name_list.append(args['tra_lbl_dir'] + imgIdx + args['label_ext'])

    print('**********************************************')
    print('train images: ', len(tra_img_name_list))
    print('train labels: ', len(tra_lbl_name_list))
    print('**********************************************')

    train_num = len(tra_img_name_list)
    salobj_dataset = SalObjDataset(img_name_list=tra_img_name_list, lbl_name_list=tra_lbl_name_list,
                                   transform=transforms.Compose([RescaleT(256), RandomCrop(224), ToTensorLab(flag=0)]))

    salobj_dataloader = DataLoader(salobj_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['workers'])

    # ------- 3. define model --------
    # define the net
    net = EDRNet(in_channels=3)
    net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.RMSprop(net.parameters(), lr=args['lr'], alpha=0.9)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    net.train()

    for epoch in range(args['epoch']):

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num+1
            ite_num4val = ite_num4val + 1
            inputs_v, labels_v = data['image'], data['label']
            inputs_v = inputs_v.type(torch.FloatTensor)
            labels_v = labels_v.type(torch.FloatTensor)

            inputs_v = inputs_v.cuda()
            labels_v = labels_v.cuda()

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            s_out, s0, s1, s2, s3, s4, sb = net(inputs_v)
            #
            loss2, loss = muti_loss_fusion(s_out, s0, s1, s2, s3, s4, sb, labels_v)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data
            running_tar_loss += loss2.data

            # del temporary outputs and loss
            del s_out, s0, s1, s2, s3, s4, sb, loss2, loss
            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, predicted loss: %3f " % (
                    epoch+1, args['epoch'], (i+1)*args['batch_size'], train_num, ite_num, running_loss/ite_num4val, running_tar_loss/ite_num4val))

        if (epoch+1) % 50 == 0:           # save model every 50 epochs
            torch.save(net.state_dict(), args['checkpoint'] + "EDRNet_epoch_%d_trnloss_%3f_priloss_%3f.pth" % (
                (epoch+1), running_loss/ite_num4val, running_tar_loss/ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            ite_num4val = 0

    print('-------------Congratulations! Training Done!!!-------------')


if __name__ == "__main__":
    main()






