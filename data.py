from torchvision import datasets, transforms
import string
import os
import random
import shutil

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.EMNIST(
    root="./data",
    split="letters",
    train=True,
    download=True,
    transform=transform
)

base_dir = "emnist_letters_folder"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

for letter in string.ascii_uppercase:
    os.makedirs(os.path.join(base_dir, letter), exist_ok=True)

for img, label in dataset:
    letter = chr(label + 64)
    img = img.squeeze(0)

    img_pil = transforms.ToPILImage()(img)

    save_path = os.path.join(base_dir, letter)
    file_name = f"{len(os.listdir(save_path))}.png"

    img_pil.save(os.path.join(save_path, file_name))

split_ratio = 0.8

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for letter in os.listdir(base_dir):
    letter_path = os.path.join(base_dir, letter)

    if not os.path.isdir(letter_path):
        continue

    if letter in ["train", "test"]:
        continue

    images = os.listdir(letter_path)

    random.shuffle(images)

    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    test_images = images[split_index:]

    os.makedirs(os.path.join(train_dir, letter), exist_ok=True)
    os.makedirs(os.path.join(test_dir, letter), exist_ok=True)

    for img in train_images:
        shutil.move(
            os.path.join(letter_path, img),
            os.path.join(train_dir, letter, img)
        )

    for img in test_images:
        shutil.move(
            os.path.join(letter_path, img),
            os.path.join(test_dir, letter, img)
        )

    os.rmdir(letter_path)

print("Готово")