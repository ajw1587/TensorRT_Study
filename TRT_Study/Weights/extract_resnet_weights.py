import struct
import torch
from torchvision import models
from torchsummary import summary

if __name__ == '__main__':
    resnet18 = models.resnet18(pretrained=True)
    resnet18.to('cuda:0')
    resnet18.eval()
    summary(resnet18, (3, 224, 224))

    input = torch.ones(1, 3, 224, 224).to('cuda:0')
    out = resnet18(input)
    print("test")

    # # Weights Extraction
    # f = open('resnet18.wts', 'w')
    # f.write('{}\n'.format(len(resnet18.state_dict().keys())))
    # for k, v in resnet18.state_dict().items():
    #     print('key: {}      value: {}'.format(k, v.shape))
    #     vr = v.reshape(-1).cpu().numpy()
    #     f.write('{} {}'.format(k, len(vr)))
    #     for vv in vr:
    #         f.write(' ')
    #         f.write(struct.pack(">f", float(vv)).hex())
    #     f.write('\n')
