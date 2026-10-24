import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ============================================================
# 任务 2：使用 CIFAR-10（彩色 3 通道，32x32）
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    # CIFAR-10 是 3 通道，Normalize 要给 3 个均值/标准差
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10(
    root='./data',train=True,
    download=True, transform=transform)

test_data  = datasets.CIFAR10(
    root='./data', train=False, 
    download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

# ============================================================
# 任务 1：基础 CNN 模型
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一层卷积：in_channels=3（彩色图，这是任务2的踩坑点）
        # out_channels=32, kernel_size=3, padding=1 保证尺寸不变
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 每次池化尺寸减半
        # 第二层卷积：32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 32x32 经过两次 pool 变成 8x8，通道 64
        # 所以全连接输入是 64 * 8 * 8
        self.d = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 类
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # -> [B, 32, 16, 16]
        x = self.pool(self.relu(self.conv2(x)))  # -> [B, 64, 8, 8]
        x = x.view(x.size(0), -1)                # 展平 -> [B, 64*8*8]
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============================================================
# 训练准备
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter('runs/cnn_cifar10')  # TensorBoard 日志目录

# ============================================================
# 训练循环
# ============================================================
epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        
        pred = model(images)
        loss = loss_fn(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        running_loss += loss.item()#.item让tensor变成数据
        _, predicted = pred.max(1)
        #每一个类别都有对应的分数，选择分数里面最高的作为类别，_作为占位符接收答案
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        #.eq表示pred和labels进行比较/.sum表示这一批图的0和1加起来（True=1)


    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total

    # 验证
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / len(test_loader)
    val_acc = 100. * val_correct / val_total

    # 写入 TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

writer.close()