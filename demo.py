import sys
from dot import HSIDataset, Encoder,  Decoder, DOT, SpectralClustering
import torch
import numpy as np
import ot

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # load img and gt

    root = './HSI_Datasets/'

    im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    #im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    #im_, gt_ = 'PaviaU', 'PaviaU_gt'

    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    print(img_path)

    # for nb_comps in range(2, 31, 1):
    # for size in range(5, 31, 2):
    NEIGHBORING_SIZE = 13
    EPOCH = 100
    LEARNING_RATE = 0.002
    REG_LAP = 0.001  # beta
    REG_LATENT = 100.  # alpha
    WEIGHT_DECAY = 0.001  # lambda
    SEED = None  # random seed
    nb_comps = 5
    VERBOSE_TIME = 1 
    BATCH_SIZE = 1
    if im_ == 'PaviaU':
        #img, gt = img[150:350, 100:200, :], gt[150:350, 100:200]
        SEED = 33
        EPOCH = 100
        LEARNING_RATE = 0.0002

        REG_LATENT = 100   # alpha
        WEIGHT_DECAY = 0.1    # lambda

        #REG_LATENT = 100   # alpha
        #WEIGHT_DECAY = 0.1    # lambda

    if im_ == 'Indian_pines_corrected':
        #img, gt = img[30:115, 24:94, :], gt[30:115, 24:94]
        SEED = 133
        NEIGHBORING_SIZE = 13
        EPOCH = 100
        LEARNING_RATE = 0.0002
        REG_LATENT = 100.  # alpha
        WEIGHT_DECAY = 0.01  # lambda
    if im_ == 'Salinas_corrected':
        #img, gt = img[0:140, 50:200, :], gt[0:140, 50:200]
        SEED = 123
    if im_ == 'SalinasA_corrected':
        SEED = 10
        NEIGHBORING_SIZE = 9
        EPOCH = 100
        LEARNING_RATE = 0.0002
        REG_LATENT = 100.  # alpha
        WEIGHT_DECAY = 0.01  # lambda
    if im_ == 'Houston':
        #img, gt = img[:, 0:680, :], gt[:, 0:680]
        SEED = 133
        NEIGHBORING_SIZE = 9
        EPOCH = 50
        LEARNING_RATE = 0.0002
        REG_LAP = 0.001  # beta
        REG_LATENT = 100.  # alpha
        WEIGHT_DECAY = 0.001  # lambda
        VERBOSE_TIME = 10


    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_data = HSIDataset(img_path, gt_path, im_)
    nclusters = train_data.get_nclusters()
    nsamples = train_data.get_nsamples()
    print("nclusters: %s" % (nclusters))
 
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE,
        shuffle=False, 
        pin_memory=torch.cuda.is_available())

    Encoder = Encoder(nb_comps)
    Decoder = Decoder(nb_comps)
    model = DOT(Encoder, Decoder, nsamples).to(device)
    sc = SpectralClustering()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = model.loss_wass
   
    '''
    for data in train_loader:
        x, y = data
        _, _, m, n = x.shape
    a1 = torch.ones(m) / m
    a1 = a1.clone().detach().requires_grad_(True)
    lr=1e-2
    '''
 
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            y = y.clone().detach().squeeze().numpy()
            x = torch.squeeze(x)
            x = x.permute(0, 3, 1, 2)
            b, c, m, n = x.shape
            x = x.float().to(device)
            z, z_hat, x_hat, C_1 = model(x)
            loss = loss_func(z, z_hat, x, x_hat, C_1, REG_LATENT, WEIGHT_DECAY)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #for p in model.named_parameters():
	        #    print(p[0])
            ''''
            with torch.no_grad():
                grad = a1.grad
                a1 -= grad * lr   # step
                a1.grad.zero_()
                a1.data = ot.utils.proj_simplex(a1)
            print(a1.clone().detach().cpu().numpy())
            '''
            print('Epoch: %s, loss: %s' % (epoch, loss.data.cpu().numpy()))
        if (epoch + 1) % VERBOSE_TIME == 0:
            ro = 0.3
            _, _, _, C_2 = model(x)
            C_2 = C_2.clone().detach().cpu().numpy()
            print(C_2.min(), C_2.max())
            torch.save(model.state_dict(), './models/%s_net_epoch_%s.pkl' % (im_, epoch))
            np.save('./models/%s_C_epoch_%s.npy' % (im_, epoch), C_2)
            plt.imshow(C_2)  # spectral
            plt.savefig("./models/%s_C_epoch_%s.pdf" % (im_, epoch), format='pdf', bbox_inches='tight')


            y_pre, affinity = sc.predict(C_2, nclusters, ro)
            plt.figure()
            plt.imshow(affinity)
            plt.savefig("./models/%s_affinity_epoch_%s.pdf" % (im_, epoch), format='pdf', bbox_inches='tight')
            acc, nmi, kappa, ca = sc.cluster_accuracy(y, y_pre)
            #print('Epoch: %s, loss: %s' % (epoch, loss.data.cpu().numpy()))
            #print('Epoch: %s, loss: %s, acc:%s' % (epoch, loss.data.cpu().numpy(), (acc, nmi, kappa, ca)))
            print('Epoch: %s, loss: %s, acc:%s' % (epoch, loss.data.cpu().numpy(), (acc, nmi, kappa)))
