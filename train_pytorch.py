import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from data import train_images, train_labels, test_labels, test_images
from notes.ConvNet import ConvNet
from torch.utils.data import DataLoader, TensorDataset

class Model(object):
    FILE_PATH = "model\model.pth"  # 模型进行存储和读取的地方
    def __init__(self):
        self.model = None

    def read_trainData(self, X_train, Y_train,batch_size):
        # 读取数据保存为tensor
        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        Y_train = torch.tensor(Y_train, dtype=torch.float32)
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.train_loader = train_loader
    def build_model(self,num_classes, device):
        self.model = ConvNet(num_classes).to(device)


    def train_model(self,num_epochs,device):
        # Train the model
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        total_step = total_step = len(self.train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 5 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


    def evaluate_model(self, X_test, Y_test, device):
        print('\nTesting---------------')
        # Test the model
        self.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        Y_test = torch.tensor(Y_test, dtype=torch.float32)
        train_dataset = TensorDataset(X_test, Y_test)
        self.test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                _, test = torch.max(labels.data, 1)
                total += labels.size(0)
                correct += (predicted == test).sum().item()

            print('Test Accuracy of the model on the {} test images: {} %'.format(total,100 * correct / total))
    def save(self, file_path=FILE_PATH):
        # Save the model checkpoint
        torch.save(self.model.state_dict(), file_path)
        print('Model Saved.')


if __name__ == '__main__':
    # Hyper parameters
    num_epochs = 20
    num_classes = 10
    batch_size = 100
    learning_rate = 0.0001
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model()
    model.read_trainData(train_images.data, train_labels.data, batch_size)
    model.build_model(num_classes,device)
    model.train_model(num_epochs,device)
    model.evaluate_model(test_images.data, test_labels.data, device)
    model.save()