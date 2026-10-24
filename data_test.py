import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
#对于定义数据的预处理，改格式为Tensor同时降低像素值

trainset = torchvision.datasets.CIFAR10(
    root='./data',#保存到data文件夹里面
    train = True,#表示这是训练集
    download = True,
    transform = transform
)

testset = torchvision.datasets.CIFAR10(
    root = './data',
    train = False,
    download = True,
    transform = transform
)
#下载并且加载【测试集】

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size = 3,
    shuffle = False,#测试集不需要打乱
    num_workers = 2#win系统报错就改成0
)

print("训练集大小：",len(trainset))
print("测试集大小：",len(testset))

dataiter = iter(trainloader)
images,labels = next(dataiter)

print("图片数据的Shape:",images.shape)

print("标签数据的形状：",labels.shape)

#把tensor转成图片
def imshow(img):
    img = img/2 +0.5 #反归一化
    nping = img.numpy()
    plt.imshow(np.transpose(nping,(1,2,0)))#调整通道顺序
    plt.show()


imshow(torchvision.utils.make_grid(images[:8]))
#拼接显示八张图片
