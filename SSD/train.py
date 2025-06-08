# train.py
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision import transforms
import torch.optim as optim
from custom_dataset import CustomDataset

CLASS_NAMES = ["__background__", "car", "chair", "cup", "door", "potted plant",
               "orange", "person", "phone", "tree"]

NUM_CLASSES = len(CLASS_NAMES)
EPOCHS = 50
BATCH_SIZE = 4
LR = 0.002
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

dataset = CustomDataset("images", "annotations", transforms=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

model = ssd300_vgg16(weights=None, weights_backbone=None, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

def train_one_epoch():
    model.train()
    total_loss = 0

    for images, targets in dataloader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(dataloader)

for epoch in range(EPOCHS):
    avg_loss = train_one_epoch()
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "ssd300_custom.pth")
print("Model saved")
