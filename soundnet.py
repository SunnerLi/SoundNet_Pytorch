from torch.autograd import Variable
import torch.nn as nn
import torch

"""
    This script defines the structure of SouneNet

    @reference: https://github.com/EsamGhaleb/soundNet_pytorch
"""

class SoundNet(nn.Module):
    def __init__(self):
        """
            The constructor of SoundNet
        """
        super().__init__()
    
        # Conv-1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(2, 1), padding=(32, 0))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1)) 

        # Conv-2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1), padding=(16, 0))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1)) 

        # Conv-3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1), padding=(8, 0))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU(True)  

        # Conv-4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1), padding=(4, 0))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU(True)  

        # Conv-5
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU(True)
        self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1)) 

        # Conv-6
        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.relu6 = nn.ReLU(True)  

        # Conv-7
        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        self.relu7 = nn.ReLU(True)  

        # Conv-8
        self.conv8_objs = nn.Conv2d(1024, 1000, kernel_size=(8, 1), stride=(2, 1))
        self.conv8_scns = nn.Conv2d(1024, 401, kernel_size=(8, 1), stride=(2, 1))

    def forward(self, waveform):
    	"""
            The forward process of SoundNet

    		Arg:    waveform     (torch.autograd.Variable)   - Raw 20s waveform.
            Ret:    The list of each layer's output
    	"""    
    	out1 = self.maxpool1(self.relu1(self.batchnorm1(self.conv1(waveform))))
    	out2 = self.maxpool2(self.relu2(self.batchnorm2(self.conv2(out1))))
    	out3 = self.relu3(self.batchnorm3(self.conv3(out2)))
    	out4 = self.relu4(self.batchnorm4(self.conv4(out3)))
    	out5 = self.maxpool5(self.relu5(self.batchnorm5(self.conv5(out4))))
    	out6 = self.relu6(self.batchnorm6(self.conv6(out5)))
    	out7 = self.relu7(self.batchnorm7(self.conv7(out6)))
    	snds = self.conv8_objs(out7)    
    	scns = self.conv8_scns(out7)    
    	return [out1, out2, out3, out4, out5, out6, out7, [snds, scns]]

if __name__ == '__main__':
    model = SoundNet()
    model.load_state_dict(torch.load('sound8.pth'))