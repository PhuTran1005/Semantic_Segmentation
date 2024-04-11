import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset


class Carvana(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """Implementation of Carvana DataLoader

        Args:
            image_dir (str): image directory
            mask_dir (str): mask directory
            transform (optional): transform operations. Defaults to None.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.list_images = os.listdir(image_dir)

    def __len__(self):
        """Default __len__ function

        Returns:
            int: Number of samples in dataset
        """
        return len(self.list_images)
    
    def __getitem__(self, idx):
        """Default __getitem__ function

        Args:
            idx (int): intdex of sample in dataset

        Returns:
            dict: image correspondings to mask
        """
        img_path = os.path.join(self.image_dir, self.list_images[idx])
        mask_path = os.path.join(self.mask_dir, self.list_images[idx].replace('.jpg', '_mask.gif'))
        
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return {'image': image, 'mask': mask}


# test dataloader
if __name__ == '__main':
    from torch.utils.data import Dataloader


    train_dataloader = Dataloader(
        Carvana(
            'dataset/train/images',
            'dataset/train/mask',
            transform=None),
        shuffle=True,
        batch_size=32
    )

    test_dataloader = Dataloader(
        Carvana(
            'dataset/test/images',
            'dataset/test/mask',
            transform=None),
        shuffle=False,
        batch_size=1
    )

    # iterate through the Dataloader
    train_samples = next(iter(train_dataloader))
    print(f'Train images batch shape: {train_samples['image'].shape}')
    print(f'Test masks batch shape: {train_samples['mask'].shape}')

    test_samples = next(iter(test_dataloader))
    print(f'Train images batch shape: {test_samples['image'].shape}')
    print(f'Test masks batch shape: {test_samples['mask'].shape}')
