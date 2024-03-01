import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random

class UnalignedTripletDataset(BaseDataset):


    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        # self.opt = opt
        # self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot + '/', opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot + '/', opt.phase + 'B')

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.input_nc
        output_nc = self.opt.output_nc
        #self.transform = get_transform(opt)
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        #A = self.transform(A_img)
        #B = self.transform(B_img)
	    # get the triplet from A
        A_img = A_img.resize((self.opt.load_size * 3, self.opt.load_size), Image.BICUBIC)
        A_img = self.transform(A_img)

        w_total = A_img.size(2)
        w = int(w_total / 3)
        h = A_img.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.crop_size - 1))
        h_offset = random.randint(0, max(0, h - self.opt.crop_size - 1))

        A0 = A_img[:, h_offset:h_offset + self.opt.crop_size,
                w_offset:w_offset + self.opt.crop_size]

        A1 = A_img[:, h_offset:h_offset + self.opt.crop_size,
               w + w_offset:w + w_offset + self.opt.crop_size]

        A2 = A_img[:, h_offset:h_offset + self.opt.crop_size,
               2*w + w_offset :2*w + w_offset + self.opt.crop_size]

	    ## -- get the triplet from B
        B_img = B_img.resize((self.opt.load_size * 3, self.opt.load_size), Image.BICUBIC)
        B_img = self.transform(B_img)

        w_total = B_img.size(2)
        w = int(w_total / 3)
        h = B_img.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.crop_size - 1))
        h_offset = random.randint(0, max(0, h - self.opt.crop_size - 1))

        B0 = B_img[:, h_offset:h_offset + self.opt.crop_size,
                w_offset:w_offset + self.opt.crop_size]

        B1 = B_img[:, h_offset:h_offset + self.opt.crop_size,
               w + w_offset:w + w_offset + self.opt.crop_size]

        B2 = B_img[:, h_offset:h_offset + self.opt.crop_size,
               2*w + w_offset :2*w + w_offset + self.opt.crop_size]


        return {'A0': A0, 'A1': A1, 'A2': A2, 'B0': B0, 'B1': B1, 'B2': B2,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedTripletDataset'
