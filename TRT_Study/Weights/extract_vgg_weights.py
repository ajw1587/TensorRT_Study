import torch
import torchvision.models as models
import struct
from torchsummary import summary


def main():
    print('cuda device count: ', torch.cuda.device_count())
    # net = torch.load('vgg.pth')
    net = models.vgg11(pretrained=True)
    net = net.to('cuda:0')
    net = net.eval()
    print('model: ', net)
    # print('state dict: ', net.state_dict().keys())
    tmp = torch.ones(1, 3, 224, 224).to('cuda:0')
    print('input: ', tmp)
    out = net(tmp)

    print('output:', out)

    summary(net, (3, 224, 224))
    # return
    f = open("test.wts", 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k, v in net.state_dict().items():
        print('key: {}      value: {}'.format(k, v.shape))
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")


if __name__ == '__main__':
    main()
    # 출처 : https://github.com/wang-xinyu/pytorchx/blob/master/vgg/inference.py
