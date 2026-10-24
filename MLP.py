import torch
from torch import nn#神经网络工具
from torch.utils.data import DataLoader#数据加载功能
from torchvision import datasets
from torchvision.transforms import v2#数据转换功能，转成可以理解的格式

#引入TensorBoard
from torch.utils.tensorboard import SummaryWriter
import os

from torch.utils.data import DataLoader, random_split

train_writer = SummaryWriter(log_dir=os.path.join('logs', 'train'))
val_writer = SummaryWriter(log_dir=os.path.join('logs', 'val'))

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)#ToImage 数据格式变成图片格式
#ToDtype把颜色数据转化成数字格式，scale表示颜色值的范围统一缩小
#这里的FashionMNIST就是固定的下载顺序，下载前面六千张作为训练库
#target_transform的意思是把标签答案改掉

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

#开始“发牌”
# batch表示一次性发多少内容给AI（自动批处理）
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#检查一下发送的内容
for X, y in test_dataloader:
    #x是题目；y是答案
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    #Number（多少张）/channel（1黑白，3彩色）/height/width
    print(f"Shape of y: {y.shape} {y.dtype}")
    break#抽查一下就行，太多了内存会爆

#选择需要使用的device并且打印出来
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):#初始化函数
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),#变成512个特征点
            nn.ReLU(),#过滤信息，负数变成0
            nn.Linear(512, 512),#再一次用512种方法提炼
            nn.ReLU(),
            nn.Linear(512, 10)#最后把特征浓缩成10个数字（由于数据库里面有十种类别）
        )

    def forward(self, x):
        x = self.flatten(x)#拍扁
        logits = self.linear_relu_stack(x)#进入流水线获得十个得分
        return logits #把得分返回

model = NeuralNetwork().to(device)#造出大脑以后放到发动机/加速器上面
print(model)#打印模型内部结构


#优化参数过程

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#loss_fn：损失函数/optimizer：优化器调整参数/lr学习率：学习率适中，太慢效率低，太快容易错过答案

#定义训练过程

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)#数一下总共多少张照片
    model.train()#切换学习模式
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)#搬到显卡里面计算

        # Compute prediction error
        pred = model(X)#模型预测结果
        loss = loss_fn(pred, y)#错误预测值

        # Backpropagation
        loss.backward()#反向传播，找哪里出现问题，算出计算梯度
        optimizer.step()#优化器进行调整参数
        optimizer.zero_grad()#把错题本清空？
        

#每次进行100次循环就报告一下错误率，学习进度
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#进行测试
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)#总共多少照片，多少次循环
    num_batches = len(dataloader)
    model.eval()#切换模式
    test_loss, correct = 0, 0#计分板
    #不允许考试改错
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            #pred.argmax用来选择分数最高的那个预测答案
    test_loss /= num_batches
    correct /= size#正确率
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#进行5轮大循环，次数过多可能过拟合
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

for epoch in range(epochs):
   # 训练阶段
   train_loss = 0.0
   model.train()
   for inputs, labels in train_dataloader:
       inputs = inputs.to(device)
       labels = labels.to(device)
       optimizer.zero_grad()
       outputs = model(inputs)
       loss = loss_fn(outputs, labels)
       loss.backward()
       optimizer.step()
       train_loss += loss.item()
   # 验证阶段
   val_loss = 0.0
   model.eval()
   with torch.no_grad():
       for inputs, labels in test_dataloader:
           inputs = inputs.to(device)
           labels = labels.to(device)
           outputs = model(inputs)
           loss = loss_fn(outputs, labels)
           val_loss += loss.item()
   # 写入 TensorBoard
   train_writer.add_scalar('Loss', train_loss / len(train_dataloader), epoch)
   val_writer.add_scalar('Loss', val_loss / len(test_dataloader), epoch)

#保存模型
torch.save(model.state_dict(), "model.pth")
#以字典的形式保存 .pth是pytorch专用的文件后缀
#并且保存的只是调整出来的参数，而没有骨架/需要再次使用的时候更加容易
#再一次model = NeuralNetwork().to(device)
#model.load_state_dict(torch.load("model.pth", weights_only=True))
print("Saved PyTorch Model State to model.pth")