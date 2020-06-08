# coding=utf-8

from skimage import io
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model.EDRNet import EDRNet
import glob
import timeit


def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)
	dn = (d-mi)/(ma-mi)
	return dn


def save_output(image_name, pred, d_dir):
	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()
	im = Image.fromarray(predict_np*255).convert('RGB')
	image = io.imread(image_name)
	imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
	img_name = image_name.split("/")[-1]       # ubuntu
	imidx = img_name.split(".")[0]
	imo.save(d_dir+imidx+'.png')


def main():
	# --------- 1. get image path and name ---------

	image_dir = './Data/imgs/'
	prediction_dir = './Data/test_results/'
	model_dir = './trained_models/EDRNet_epoch_600.pth'     # path of pre-trained model
	img_name_list = glob.glob(image_dir + '*.bmp')

	# --------- 2. dataloader ---------
	test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
										transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)]))
	test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=5)

	# --------- 3. model define ---------
	print("...load EDRNet...")
	net = EDRNet(in_channels=3)
	net.load_state_dict(torch.load(model_dir))
	net.cuda()

	net.eval()

	start = timeit.default_timer()
	# --------- 4. inference for each image ---------
	with torch.no_grad():
		for i_test, data_test in enumerate(test_salobj_dataloader):
			print("inferencing:", img_name_list[i_test].split("/")[-1])
			inputs_test = data_test['image']
			inputs_test = inputs_test.type(torch.FloatTensor)
			inputs_test = inputs_test.cuda()

			s_out, s0, s1, s2, s3, s4, sb = net(inputs_test)

			# normalization
			pred = s_out[:, 0, :, :]
			pred = normPRED(pred)

			# save results to test_results folder
			save_output(img_name_list[i_test], pred, prediction_dir)
			del s_out, s0, s1, s2, s3, s4, sb

	end = timeit.default_timer()
	print(str(end-start))


if __name__ == "__main__":
	main()

