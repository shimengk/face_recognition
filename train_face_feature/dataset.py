from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class MyDataset(Dataset):

    def __init__(self, main_dir):

        self.dataset = []
        for idx, face_dir in enumerate(os.listdir(main_dir)):
            for face_filename in os.listdir(os.path.join(main_dir, face_dir)):
                self.dataset.append([os.path.join(main_dir, face_dir, face_filename), int(idx)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        img_data = tf(Image.open(data[0]))
        return img_data, data[1]


if __name__ == '__main__':
    mydataset = MyDataset("data3")
    print(mydataset[10])
    print(mydataset[10][0].shape)
