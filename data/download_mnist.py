import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 指定下载路径
download_path = "./mnist"

# 下载MNIST训练集和测试集
train_dataset = datasets.MNIST(root=download_path, train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root=download_path, train=False, download=True, transform=transforms.ToTensor())

print("MNIST数据集已下载到:", download_path)