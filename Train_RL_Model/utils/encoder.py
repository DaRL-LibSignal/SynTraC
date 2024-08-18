import torch
import torch.nn as nn
import d3rlpy
import dataclasses
from torchvision.models import resnet50, ResNet50_Weights, resnet, ResNet, vit_h_14, ViT_H_14_Weights, VisionTransformer


class PreTrainedFourWayVisonModel(nn.Module):
	def __init__(self, obs_shape, feature_size):
		super().__init__()
		self.feature_size = feature_size
		self.obs_shape = obs_shape
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=10,
			kernel_size=(5, 5))
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
		# initialize second set of CONV => RELU => POOL layers
		self.conv2 = nn.Conv2d(in_channels=10, out_channels=30,
			kernel_size=(5, 5))
		self.relu2 = nn.ReLU()
		self.maxpool2 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
		# initialize first (and only) set of FC => RELU layers
		f_dim = self._get_conv_output(obs_shape[1:])
		self.viewA1 = nn.Flatten()
		self.viewA2 = nn.Linear(in_features=f_dim, out_features=self.feature_size)
		self.viewB1 = nn.Flatten()
		self.viewB2 = nn.Linear(in_features=f_dim, out_features=self.feature_size)
		self.viewC1 = nn.Flatten()
		self.viewC2 = nn.Linear(in_features=f_dim, out_features=self.feature_size)
		self.viewD1 = nn.Flatten()
		self.viewD2 = nn.Linear(in_features=f_dim, out_features=self.feature_size)
		self._test_propagation(self.obs_shape)
		
		# self.fc1 =nn.Linear(in_features=800, out_features=500)
		# self.relu3 = ReLU()
	def _forward_features(self, x):
		x = self.maxpool1(self.relu1(self.conv1(x)))
		x = self.maxpool2(self.relu2(self.conv2(x)))
		return x

	def _get_conv_output(self, shape):
		bs = 1
		input = nn.Parameter(torch.rand(bs, *shape))
		output_feature = self._forward_features(input)
		n_size = output_feature.data.view(bs, -1).size(1)
		return n_size

	def forward(self, x):
		x = x.reshape(-1, self.obs_shape[-3], self.obs_shape[-2], self.obs_shape[-1])
		x = self.maxpool1(self.relu1(self.conv1(x)))
		x = self.maxpool2(self.relu2(self.conv2(x)))
		x = x.reshape(-1, self.obs_shape[-4], x.shape[-3], x.shape[-2], x.shape[-1])
		x_1 = self.viewA2(self.viewA1(x[:, 0 , :, :, :]))
		x_2 = self.viewB2(self.viewB1(x[:, 1 , :, :, :]))
		x_3 = self.viewC2(self.viewC1(x[:, 2 , :, :, :]))
		x_4 = self.viewD2(self.viewD1(x[:, 3 , :, :, :]))
		return torch.concatenate((x_1, x_2, x_3, x_4), axis=1)
	
	def _test_propagation(self, shape):
		bs = 32
		input = nn.Parameter(torch.rand(bs, *shape))
		x = input.reshape(-1, shape[-3], shape[-2], shape[-1])
		x = self.conv1(x)
		x = self.relu2(x)
		x = self.maxpool1(x)
		x = self.maxpool2(self.relu2(self.conv2(x)))
		x = x


class FourWayVisonModel(nn.Module):
	def __init__(self, obs_shape, feature_size):
		super().__init__()
		self.feature_size = feature_size
		self.obs_shape = obs_shape
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=10,
			kernel_size=(5, 5))
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
		# initialize second set of CONV => RELU => POOL layers
		self.conv2 = nn.Conv2d(in_channels=10, out_channels=30,
			kernel_size=(5, 5))
		self.relu2 = nn.ReLU()
		self.maxpool2 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
		# initialize first (and only) set of FC => RELU layers
		f_dim = self._get_conv_output(obs_shape[1:])
		self.viewA1 = nn.Flatten()
		self.viewA2 = nn.Linear(in_features=f_dim, out_features=self.feature_size)
		self.viewB1 = nn.Flatten()
		self.viewB2 = nn.Linear(in_features=f_dim, out_features=self.feature_size)
		self.viewC1 = nn.Flatten()
		self.viewC2 = nn.Linear(in_features=f_dim, out_features=self.feature_size)
		self.viewD1 = nn.Flatten()
		self.viewD2 = nn.Linear(in_features=f_dim, out_features=self.feature_size)
		self._test_propagation(self.obs_shape)
		
		# self.fc1 =nn.Linear(in_features=800, out_features=500)
		# self.relu3 = ReLU()
	def _forward_features(self, x):
		x = self.maxpool1(self.relu1(self.conv1(x)))
		x = self.maxpool2(self.rePreTrainedModifiedResNetlu2(self.conv2(x)))
		return x

	def _get_conv_output(self, shape):
		bs = 1
		input = nn.Parameter(torch.rand(bs, *shape))
		output_feature = self._forward_features(input)
		n_size = output_feature.data.view(bs, -1).size(1)
		return n_size

	def forward(self, x):
		x = x.reshape(-1, self.obs_shape[-3], self.obs_shape[-2], self.obs_shape[-1])
		x = self.maxpool1(self.relu1(self.conv1(x)))
		x = self.maxpool2(self.relu2(self.conv2(x)))
		x = x.reshape(-1, self.obs_shape[-4], x.shape[-3], x.shape[-2], x.shape[-1])
		x_1 = self.viewA2(self.viewA1(x[:, 0 , :, :, :]))
		x_2 = self.viewB2(self.viewB1(x[:, 1 , :, :, :]))
		x_3 = self.viewC2(self.viewC1(x[:, 2 , :, :, :]))
		x_4 = self.viewD2(self.viewD1(x[:, 3 , :, :, :]))
		return torch.concatenate((x_1, x_2, x_3, x_4), axis=1)
	
	def _test_propagation(self, shape):
		bs = 32
		input = nn.Parameter(torch.rand(bs, *shape))
		x = input.reshape(-1, shape[-3], shape[-2], shape[-1])
		x = self.conv1(x)
		x = self.relu2(x)
		x = self.maxpool1(x)
		x = self.maxpool2(self.relu2(self.conv2(x)))
		x = x

class FourWayPreTrainedVisonModel(nn.Module):
	def __init__(self, obs_shape, feature_size):
		super().__init__()
		self.feature_size = feature_size
		self.obs_shape = obs_shape
		# initialize first (and only) set of FC => RELU layers
		# f_dim = self._get_conv_output(obs_shape[1:])
		self.feature_extractor = PreTrainedModifiedResNet()
		f_dim = self._get_conv_output(obs_shape[1:])
		self.viewA1 = nn.Flatten()
		self.viewA2 = nn.Linear(in_features=f_dim, out_features=self.feature_size)
		self.viewB1 = nn.Flatten()
		self.viewB2 = nn.Linear(in_features=f_dim, out_features=self.feature_size)
		self.viewC1 = nn.Flatten()
		self.viewC2 = nn.Linear(in_features=f_dim, out_features=self.feature_size)
		self.viewD1 = nn.Flatten()
		self.viewD2 = nn.Linear(in_features=f_dim, out_features=self.feature_size)
		# self._test_propagation(self.obs_shape)
		print()

	def _forward_features(self, x):
		x = self.feature_extractor(x)
		return x

	def _get_conv_output(self, shape):
		bs = 1
		input = nn.Parameter(torch.rand(bs, *shape))
		output_feature = self._forward_features(input)
		n_size = output_feature.data.view(bs, -1).size(1)
		return n_size
	
	def forward(self, x):
		x = x.reshape(-1, self.obs_shape[-3], self.obs_shape[-2], self.obs_shape[-1])
		x = self.feature_extractor(x)
		x = x.reshape(-1, self.obs_shape[-4], x.shape[-3], x.shape[-2], x.shape[-1])
		x_1 = self.viewA2(self.viewA1(x[:, 0 , :, :, :]))
		x_2 = self.viewB2(self.viewB1(x[:, 1 , :, :, :]))
		x_3 = self.viewC2(self.viewC1(x[:, 2 , :, :, :]))
		x_4 = self.viewD2(self.viewD1(x[:, 3 , :, :, :]))
		return torch.concatenate((x_1, x_2, x_3, x_4), axis=1)

		
class PreTrainedModifiedResNet(ResNet):
	def __init__(self):
		super(PreTrainedModifiedResNet, self).__init__(resnet.Bottleneck, [3, 4, 6, 3])
		# print(self.layer1[0].conv1.weight[0][0])
# 		self.load_state_dict(ResNet50_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True))
		self.load_state_dict(ResNet50_Weights.DEFAULT.get_state_dict(progress=True))
		# print(self.layer1[0].conv1.weight[0][0])
		# print(self.layer1[0].conv1.param())
		self.fc = None
		
		# You can also add new layers, change connections, etc.
		self._frozen()
		
	def _frozen(self):
		for param in self.parameters():
			param.requires_grad = False
			
	def _forward_impl(self, x):
		# See note [TorchScript super()]
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		return x

@dataclasses.dataclass()
class CustomEncoderFactory(d3rlpy.models.EncoderFactory):
	feature_size: int
	
	def create(self, observation_shape):
		return FourWayPreTrainedVisonModel(observation_shape, self.feature_size)
	
	@staticmethod
	def get_type() -> str:
		return "custom"
	

	
