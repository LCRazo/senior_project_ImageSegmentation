import torch
from torch.utils.data import Dataset
import numpy as np
from os.path import join


class TrainMosDataset(Dataset):

    def __init__(self, train_list, datadir=None, args=None, transforms=None):
        #construct the paths to the directories containing images
        img_dir = join(datadir, 'imgs')
        label_dir = join(datadir, 'masks')
        #initializes empty list to store names of the files in dataset
        names = []
        #Opens text file containing list of file names for reading
        with open(join(datadir, train_list + '.txt')) as f:
            #iterates each line
            for line in f:
                #removes leading and trailing whitespace
                line = line.strip()
                #assignes stripped line to the variable
                name = line
                #adds file name to the names list
                names.append(name)

        # print('file list:' + str(names))
        #assigns values computed above to instance variables of class
        self.names = names
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.args = args
        self.transforms = transforms


    def __getitem__(self, index):
        # loads image (x) and label (y) by joining directory path with filename
        # loads label data from file specified, reads it and return content
        #ensures data is loaded and interpreted correctly with data type
        y = np.array(np.load(join(self.label_dir, self.names[index])), dtype='uint8', order='C')
        x = np.array(np.load(join(self.img_dir, self.names[index])), dtype='float32', order='C')
        # print(x.shape, y.shape)#(512, 512) (512, 512)

        #applies transformation through preprocessing (eg. normalization,rezing or dataaugumentation)
        x, y = self.transforms([x, y])

        #Adds a singleton dimension often needed for grayscale images
        x = x[..., None]        # (513, 513, 1)

        #repeats singleton dimension three times to create 3-channel image and transposes channel to the front
        x = np.ascontiguousarray(x.repeat(3, 2).transpose(2, 0, 1))  # (3, 513, 513)

        #improves performance during training by keeping y contiguous
        y = np.ascontiguousarray(y)

        #converts numPys arrays to pyTorch tensors(required for training neural networks)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # (3, 513, 513) (513, 513)

        #transformed and prepared images are returned as tuple
        return x, y

    def __len__(self):
        #returns total number of samples in dataset
        return len(self.names)


class TrainCovid100Dataset_1(Dataset):
    def __init__(self, train_list, datadir=None, args=None, transforms=None):
        img_dir = join(datadir, 'imgs')
        label_dir = join(datadir, 'masks1')
        names = []
        with open(join(datadir, train_list + '.txt')) as f:
            for line in f:
                line = line.strip()
                name = line
                names.append(name)
        # print('file list:' + str(names))
        self.names = names
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.args = args
        self.transforms = transforms

    def __getitem__(self, index):
        y = np.array(np.load(join(self.label_dir, self.names[index])), dtype='uint8', order='C')
        x = np.array(np.load(join(self.img_dir, self.names[index])), dtype='float32', order='C')
        # print(x.shape, y.shape)#(512, 512) (512, 512)
        x, y = self.transforms([x, y])
        x = x[..., None]        # (513, 513, 1)

        x = np.ascontiguousarray(x.repeat(3, 2).transpose(2, 0, 1))  # (3, 513, 513)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # (3, 513, 513) (513, 513)

        return x, y

    def __len__(self):
        return len(self.names)

class TrainCovid100Dataset_3(Dataset):
    def __init__(self, train_list, datadir=None, args=None, transforms=None):
        img_dir = join(datadir, 'imgs')
        label_dir = join(datadir, 'masks')
        names = []
        with open(join(datadir, train_list + '.txt')) as f:
            for line in f:
                line = line.strip()
                name = line
                names.append(name)
        # print('file list:' + str(names))
        self.names = names
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.args = args
        self.transforms = transforms

    def __getitem__(self, index):
        y = np.array(np.load(join(self.label_dir, self.names[index])), dtype='uint8', order='C')
        x = np.array(np.load(join(self.img_dir, self.names[index])), dtype='float32', order='C')
        # print(x.shape, y.shape)#(512, 512) (512, 512)
        x, y = self.transforms([x, y])
        x = x[..., None]        # (513, 513, 1)

        x = np.ascontiguousarray(x.repeat(3, 2).transpose(2, 0, 1))  # (3, 513, 513)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # (3, 513, 513) (513, 513)

        return x, y

    def __len__(self):
        return len(self.names)


class TrainCovid20Dataset(Dataset):
    def __init__(self, train_list, datadir=None, args=None, transforms=None):
        img_dir = join(datadir, 'imgs')
        label_dir = join(datadir, 'masks')
        names = []
        with open(join(datadir, train_list + '.txt')) as f:
            for line in f:
                line = line.strip()
                name = line
                names.append(name)
        # print('file list:' + str(names))
        self.names = names
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.args = args
        self.transforms = transforms

    def __getitem__(self, index):
        y = np.array(np.load(join(self.label_dir, self.names[index])), dtype='uint8', order='C')
        x = np.array(np.load(join(self.img_dir, self.names[index])), dtype='float32', order='C')
        # print('shape1:', x.shape, y.shape)
        # print(x.shape, y.shape)#(512, 512) (512, 512)
        x, y = self.transforms([x, y])
        x = x[..., None]        # (513, 513, 1)

        x = np.ascontiguousarray(x.repeat(3, 2).transpose(2, 0, 1))  # (3, 513, 513)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # print('shape2:', x.shape, y.shape)
        # (3, 513, 513) (513, 513)

        return x, y

    def __len__(self):
        return len(self.names)

