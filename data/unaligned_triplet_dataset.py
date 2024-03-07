import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageFile
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

class UnalignedTripletDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot + '/', opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot + '/', opt.phase + 'B')

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.input_nc if btoA else self.opt.input_nc
        output_nc = self.opt.output_nc if btoA else self.opt.output_nc
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))


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

        h = A_img.size[1]
        w_total = A_img.size[0]
        w = int(w_total / 3)

        A0 = A_img.crop((0, 0, w, h))
        A1 = A_img.crop((w, 0, 2*w, h))
        A2 = A_img.crop((2*w, 0, w_total, h))

        A0 = self.transform_A(A0)
        A1 = self.transform_A(A1)
        A2 = self.transform_A(A2)

        h = B_img.size[1]
        w_total = B_img.size[0]
        w = int(w_total / 3)

        B0 = B_img.crop((0, 0, w, h))
        B1 = B_img.crop((w, 0, 2*w, h))
        B2 = B_img.crop((2*w, 0, w_total, h))

        B0 = self.transform_B(B0)
        B1 = self.transform_B(B1)
        B2 = self.transform_B(B2)

        # because during tarining _1 & _2 -> _0
        return {'A0': A2, 'A1': A0, 'A2': A1, 'B0': B2, 'B1': B0, 'B2': B1,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedTripletDataset'
