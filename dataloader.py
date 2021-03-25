import numpy as np
import torch
from torchvision import transforms
from torch.utils import data
from PIL import Image

def rreplace(s, old, new):
    li = s.rsplit(old, 1)
    return new.join(li)

class Dataset:
    def __init__(self, attribute_path, image_path, mode='train', sup_train=120, sup_val=28, unsup=968, char_class=52,
                 n_style=4):
        self.attribute_path = attribute_path
        self.image_path = image_path
        self.char_class = char_class  # n_chars in one class
        self.sup_train = sup_train    # supervised n_classes in train
        self.sup_val = sup_val        # supervised n_classes in val
        self.unsup = unsup            # unsupervised n_classes
        self.n_style = n_style        # number of image sent to image encoder (same class as source - different chars)

        self.transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.mode = mode

        self.idx2attr = None
        self.attr2idx = None
        self.idx2filename = None
        self.filename2idx = None

        self.data, self.unsup_data, self.sup_data, self.val_data = self.get_data(self.attribute_path)

        self.val_map = {}
        for font in range(self.sup_val):
            self.val_map[font] = np.random.randint(0, self.unsup - 1)


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

        data = [item for item in zip(filenames, chars, attributes)]   # full_filename, number before .png, list of 0..1
        unsup_data = data[(self.sup_train + self.sup_val) * self.char_class:]
        sup_data = data[:self.sup_train * self.char_class]
        val_data = data[self.sup_train * self.char_class: (self.sup_train + self.sup_val) * self.char_class]
        data = np.concatenate([sup_data, unsup_data])
        return data, unsup_data, sup_data, val_data

    def __getitem__(self, index):
        if self.mode == 'train':
            source = self.data[index] # supervised or unsupervised
            source_label = 1 if index < self.sup_train * self.char_class else 0
            src_embed = self.unsup if index < self.sup_train * self.char_class else (index // self.char_class -
                                                                                     self.sup_train)

            p = np.random.random()
            if p < 0.5:
                # Supervised
                target_label = 1
                target_index = index % self.char_class + self.char_class * np.random.randint(0, self.sup_train - 1)  # same letter from random class
                target = self.sup_data[target_index]
                trg_embed = self.unsup
            else:
                # Unsupervised
                target_label = 0
                new_font = np.random.randint(0, self.unsup - 1)
                target_index = index % self.char_class + self.char_class * new_font
                target = self.unsup_data[target_index]
                trg_embed = new_font

        elif self.mode == 'test':
            # source from unsup train
            source_label = 0
            font_index = self.val_map[index // self.char_class]
            char_index = index % self.char_class + self.char_class * font_index
            source = self.unsup_data[char_index]
            src_embed = font_index

            # target from (sup) val
            # print('here', len(self.val_data))
            target = self.val_data[index]
            target_label = 1
            trg_embed = self.unsup

        src_filename, src_char, src_attribute = source
        trg_filename, trg_char, trg_attribute = target

        src_image = self.transforms(Image.open(self.image_path + src_filename).convert('RGB'))
        trg_image = self.transforms(Image.open(self.image_path + trg_filename).convert('RGB'))

        src_style = np.random.choice(np.array(range(10, self.char_class)), self.n_style)  # same font as source, diff chars
        src_style = [rreplace(src_filename, str(src_char), str(i)) for i in src_style]
        src_style = torch.cat([self.transforms(Image.open(self.image_path + file).convert('RGB')) for file in src_style])

        return {'src_image': src_image, 'src_char': torch.tensor(src_char),
                'src_attribute': torch.tensor(src_attribute), 'src_style': src_style,
                'src_label': source_label, 'src_embed': src_embed,
                'trg_image': trg_image, 'trg_char': torch.tensor(trg_char),
                'trg_attribute': torch.tensor(trg_attribute), 'trg_label': target_label,
                'trg_embed': trg_embed}

    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        else:
            return len(self.val_data)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    attribute_path = 'data/attributes.txt'
    image_path = 'data/image/'

    mode = 'train'

    dataset = Dataset(attribute_path, image_path, mode='test')
    dataloader = data.DataLoader(dataset=dataset,
                                  drop_last=True,
                                  batch_size=1,
                                 shuffle=True)

    for i, batch in enumerate(dataloader):
        pass
        # if i > 100:
        #     source = batch['src_image']
        #     target = batch['trg_image']
        #     image = torch.cat([source, target], dim=3)
        #     image = image.squeeze(0).permute(1, 2, 0)
        #     plt.imshow(image)
        #     plt.show()
        #     print(batch['src_char'], batch['trg_char'])

    print(k)
