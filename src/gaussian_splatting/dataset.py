from torch.utils.data import Dataset


class ImagePoseDataset(Dataset):
    def __init__(self, images_tensor, poses_tensor, masks_tensor):
        assert len(images_tensor) == len(
            poses_tensor
        ), "Number of images and poses must match"
        self.images_tensor = images_tensor
        self.poses_tensor = poses_tensor
        self.masks_tensor = masks_tensor

    def __len__(self):
        return len(self.images_tensor)

    def __getitem__(self, index):
        image = self.images_tensor[index]
        pose = self.poses_tensor[index]
        mask = self.masks_tensor[index]
        return image, pose, mask


if __name__ == "__main__":
    pass
