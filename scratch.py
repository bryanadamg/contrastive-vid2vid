import torch
from models import networks
from models.patchnce import PatchNCELoss
from options.train_options import TrainOptions
from data.unaligned_triplet_dataset import UnalignedTripletDataset


opt = TrainOptions().parse()

input_nc, output_nc = 3, 3
netG = 'resnet_9blocks'
netF = 'mlp_sample'
normG = 'instance'
ngf = 64
no_dropout = True
init_type, init_gain = 'xavier', 0.02
no_antialias, no_antialias_up = True, True
gpu_ids = -1

num_patches = 32
nce_layers = [0,4,8,12,16]
lambda_NCE = 1.0

opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1
opt.isTrain = False

netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, opt.gpu_ids, opt)
netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.gpu_ids, opt)
# netF = networks.SwinSampleF(use_swin=True)
criterionNCE = []

for nce_layer in nce_layers:
    criterionNCE.append(PatchNCELoss(opt))

def calculate_NCE_loss(src, tgt):
    n_layers = len(nce_layers)
    feat_q = netG(tgt, nce_layers, encode_only=True)
    feat_q = [feat_q[1]]
    print('feat_q_0 size: ', feat_q[0].size())

    # for i, layer in enumerate(nce_layers):
    #     print(f'feat_{layer} size: ', feat_q[i].size())

    feat_k = netG(src, nce_layers, encode_only=True)
    feat_k = [feat_k[1]]
    print('feat_k_0 size: ', feat_k[0].size())

    feat_k_pool, sample_ids = netF(feat_k, num_patches, None)

    for i, pool in enumerate(feat_k_pool):
        print(f'feat_k_pool_{i} size: ', pool.size())
        # print(f'sample_{i} size: ', sample_ids[i].size())
        
    feat_q_pool, _ = netF(feat_q, num_patches, sample_ids)

    for i, pool in enumerate(feat_q_pool):
        print(f'feat_q_pool_{i} size: ', pool.size())
        # print(f'sample_{i} size: ', sample_ids[i].size())

    print()
    print('Calculate NCE Loss')
    total_nce_loss = 0.0
    for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, criterionNCE, nce_layers):
        loss = crit(f_q, f_k) * lambda_NCE
        print('loss: ', loss.size())
        total_nce_loss += loss.mean()

    return total_nce_loss / n_layers

dataset = UnalignedTripletDataset(opt)

src = dataset[0]['A1']
tgt = dataset[0]['B1']

real = torch.cat((src, tgt), dim=0) if opt.nce_idt and opt.isTrain else src
real = real.unsqueeze(0)

real = torch.rand(2, 3, 512, 512)
# real = torch.rand(1,3,1024,1024)
# torch.Size([6, 256, 256])
print(real.size())
# if opt.flip_equivariance:
#     flipped_for_equivariance = opt.isTrain and (np.random.random() < 0.5)
#     if flipped_for_equivariance:
#         real = torch.flip(real, [3])

fake = netG(real)
print(fake.size())

loss = calculate_NCE_loss(real, fake)

print(loss)

    