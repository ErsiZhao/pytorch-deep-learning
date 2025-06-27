"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names



import random
import shutil
from pathlib import Path
from tqdm import tqdm

def split_dataset(images_dir, train_dir, test_dir, train_ratio=0.8, seed=42):
    random.seed(seed)
    images_dir = Path(images_dir)
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in tqdm(sorted(images_dir.iterdir()), desc="Processing classes"):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        train_class_dir = train_dir / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        for img in train_images:
            shutil.copy(img, train_class_dir / img.name)

        test_class_dir = test_dir / class_name
        test_class_dir.mkdir(parents=True, exist_ok=True)
        for img in test_images:
            shutil.copy(img, test_class_dir / img.name)

    print(f"\n✅ Done! Train images in '{train_dir}', test images in '{test_dir}'")

