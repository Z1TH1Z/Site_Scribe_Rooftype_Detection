import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
import random
from one_shot_model import Siamese
from one_shot_mydataset import RoofTrain

class RoofTest(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        super(RoofTest, self).__init__()
        self.dataset = dataset
        self.transform = transform

        self.class_to_imgs = {}
        for path, cls in self.dataset.imgs:
            if cls not in self.class_to_imgs:
                self.class_to_imgs[cls] = []
            self.class_to_imgs[cls].append(path)

        self.classes = list(self.class_to_imgs.keys())

        self.pairs = []
        # For robust evaluation, generate all possible pairs (including all possible same-class pairs)
        for i in range(len(self.classes)):
            for j in range(len(self.classes)):
                class_i = self.classes[i]
                class_j = self.classes[j]
                # Only create same-class pairs if there are at least 2 images
                if class_i == class_j and len(self.class_to_imgs[class_i]) < 2:
                    continue
                # For each image in class_i, pair with each image in class_j
                for img1 in self.class_to_imgs[class_i]:
                    for img2 in self.class_to_imgs[class_j]:
                        if class_i == class_j and img1 == img2:
                            continue  # don't pair image with itself
                        label = 1.0 if class_i == class_j else 0.0
                        self.pairs.append((img1, img2, label))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, img1_path, img2_path, torch.tensor([label], dtype=torch.float32)

if __name__ == '__main__':
    cuda = torch.cuda.is_available()

    # Ensure preprocessing is consistent everywhere
    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder('training')
    test_dataset = datasets.ImageFolder('evaluation')

    class RoofTrainSmall(RoofTrain):
        def __len__(self):
            return min(1000, super().__len__())

    train_set = RoofTrainSmall(train_dataset, transform=transform)
    test_set = RoofTest(test_dataset, transform=transform)

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    model = Siamese()
    if cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.BCEWithLogitsLoss()

    print("🚀 Training...\n")
    max_iter = 1000  # Increased iterations for better learning
    model.train()
    start_time = time.time()
    loss_val = 0

    for batch_id, (img1, img2, _, _, label) in enumerate(train_loader, 1):
        if batch_id > max_iter:
            break
        b_start = time.time()

        if cuda:
            img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()

        optimizer.zero_grad()
        output = model(img1, img2)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        loss_val += loss.item()

        if batch_id % 10 == 0:
            avg_loss = loss_val / 10
            print(f"[{batch_id:3d}] Avg Loss: {avg_loss:.4f} | Time: {time.time() - b_start:.2f}s")
            loss_val = 0

    print(f"\n✅ Training done in {time.time() - start_time:.2f}s\n")

    # Testing
    print("🔍 Testing...\n")
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for img1, img2, path1, path2, label in test_loader:
            if cuda:
                img1, img2 = img1.cuda(), img2.cuda()
            output = model(img1, img2)
            prob = torch.sigmoid(output).item()
            pred = prob > 0.5
            actual = label.item() > 0.5
            print(f"Pair: {path1[0]} vs {path2[0]} | Pred: {prob:.3f} | Actual: {actual}")
            if pred == actual:
                correct += 1
            total += 1

    print(f"🎯 Accuracy: {correct}/{total} = {correct / total:.2%}")
    torch.save(model.state_dict(), "siamese_model.pth")
    print("📦 Model saved as 'siamese_model.pth'")

    # Reminder for inference script
    print("\n[INFO] When using this model for inference, ensure you apply the same preprocessing steps:")
    print("- Resize to 105x105")
    print("- Convert to grayscale")
    print("- Convert to tensor")
    print("And use the same model architecture as in training.")
