import numpy as np
import torch
from PIL import Image


class Dataset:
    def __init__(self, attribute_path, image_path, transforms,
                 mode='train', sup_train=120, sup_val=28, unsup=968, char_class=52, n_style=4):
        self.attribute_path = attribute_path
        self.image_path = image_path
        self.transforms = transforms
        self.char_class = char_class
        self.sup_train = sup_train
        self.sup_val = sup_val
        self.unsup = unsup
        self.n_style = n_style

        self.mode = mode

        self.idx2attr = None
        self.attr2idx = None
        self.idx2filename = None
        self.filename2idx = None


        self.data, self.unsup_data, self.sup_data = self.get_data(self.attribute_path)


    @staticmethod
    def read_data(attribute_path):
        """
        Read attributes values from txt file

        :param attribute_path: file attributes.txt
        :return: fileds - attributes names, filenames,
                 target - character label, attributes values
        """
        x = np.genfromtxt(attribute_path, dtype=str)
        fields = x[0]
        filenames = x[1:, 0]
        chars = [int(file.split('/')[1].split('.')[0]) for file in filenames]
        attributes = x[1:, 1:].astype(np.float) / 100
        return (fields, filenames, chars, attributes)

    def get_data(self, attribute_path):
        """
        Get train/test data from txt file
        :param attribute_path: file attributes.txt
        :return: for train: data - [:120] + [140:]  (without test)
                            supervised data - [:120]
                            unsupervised data - [140:]
        """
        attr_names, filenames, chars, attributes = self.read_data(attribute_path)

        self.idx2attr = attr_names
        self.attr2idx = {a: i for i, a in enumerate(attr_names)}
        self.idx2filename = np.unique(filenames)
        self.filename2idx = {f: i for i, f in enumerate(self.idx2filename)}

        data = [item for item in zip(filenames, chars, attributes)]
        unsup_data = data[(self.sup_train + self.sup_val) * self.char_class:]
        if self.mode == 'train':
            sup_data = data[:self.sup_train * self.char_class]
        else:
            sup_data = data[self.sup_train * self.char_class: (self.sup_train + self.sup_val) * self.char_class]
        data = np.concatenate([sup_data, unsup_data])
        return data, unsup_data, sup_data


    def __getitem__(self, index):
        if self.mode == 'train':
            source = self.data[index] # supervised or unsupervised
            source_label = 1 if index < self.sup_train else 0
            p = np.random.random()
            if p < 0.5:
                # Supervised
                target_label = 1
                target_index = index % self.char_class + self.char_class * np.random.randint(0, self.sup_train - 1)
                target = self.sup_data[target_index]
            else:
                # Unsupervised
                target_label = 0
                target_index = index % self.char_class + self.char_class * np.random.randint(0, self.unsup - 1)
                target = self.unsup_data[target_index]
        elif self.mode == 'test':
            pass

        src_filename, src_char, src_attribute = source
        trg_filename, trg_char, trg_attribute = target

        src_image = self.transforms(Image.open(self.image_path + src_filename).convert('RGB'))
        trg_image = self.transforms(Image.open(self.image_path + trg_filename).convert('RGB'))

        src_style = np.random.choice(np.array(range(10, self.char_class)), self.n_style)
        src_style = [src_filename.replace(str(src_char), str(i)) for i in src_style]
        src_style = torch.cat([self.transforms(Image.open(self.image_path + file).convert('RGB')) for file in src_style])

        # fontembed? 
        return {'src_image': src_image, 'src_char': torch.tensor(src_char),
                'src_attribute': torch.tensor(src_attribute), 'src_styles': src_style,
                'src_labetl': source_label,
                'trg_image': trg_image, 'trg_char': torch.tensor(trg_char),
                'trg_attribute': torch.tensor(trg_attribute), 'trg_label': target_label}

    def __len__(self):
        return len(self.data)



if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils import data
    import matplotlib.pyplot as plt

    attribute_path = 'data/attributes.txt'
    image_path = 'data/image/'

    mode = 'train'

    transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    ])

    dataset = Dataset(attribute_path, image_path, transforms, 'train')
    dataloader = data.DataLoader(dataset=dataset,
                                  drop_last=True,
                                  batch_size=16)

    for batch in dataloader:
        img = batch['src_styles']
        print(img.shape)

        1/0
