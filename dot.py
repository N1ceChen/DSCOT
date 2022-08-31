from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
import numpy as np 

import torch
import ot
from ot.gromov import gromov_wasserstein2
import scipy.io as sio


from scipy.sparse.linalg import svds

from sklearn.preprocessing import normalize
from sklearn import cluster
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.classification import cohen_kappa_score, accuracy_score
from munkres import Munkres
class HSIDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, gt_path, im):
        nb_comps = 5
        p = Processor()
        img, gt = p.prepare_data(img_path, gt_path)
 
        if im == 'PaviaU':
            img, gt = img[150:350, 100:200, :], gt[150:350, 100:200]
            NEIGHBORING_SIZE = 13
        if im == 'Indian_pines_corrected':
            img, gt = img[30:115, 24:94, :], gt[30:115, 24:94]
            NEIGHBORING_SIZE = 13
        if im == 'SalinasA_corrected':
            NEIGHBORING_SIZE = 13
        n_row, n_column, n_band = img.shape
        img_scaled = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape(img.shape)

        # perform PCA
        pca = PCA(n_components=nb_comps)
        img = pca.fit_transform(img_scaled.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, nb_comps))
        print('pca shape: %s, percentage: %s' % (img.shape, np.sum(pca.explained_variance_ratio_)))
        x_patches, y_ = p.get_HSI_patches_rw(img, gt, (NEIGHBORING_SIZE, NEIGHBORING_SIZE))  # x_patch=(n_samples, n_width, n_height, n_band)
        print(np.unique(y_))
        # perform ZCA whitening
        # x_patches = minmax_scale(x_patches.reshape(x_patches.shape[0], -1)).reshape(x_patches.shape)
        # x_patches, _, _ = p_Cora.zca_whitening(x_patches, epsilon=10.)
        self.x = minmax_scale(x_patches.reshape(x_patches.shape[0], -1)).reshape(x_patches.shape)
        print('img shape:', img.shape)
        print('img_patches_nonzero:', self.x.shape)
        #n_samples, n_width, n_height, n_band = self.x.shape

        self.y = p.standardize_label(y_)
 
    def __len__(self):
        return 1
    def __getitem__(self, index):
        return self.x, self.y
    def get_nclusters(self):
        return np.unique(self.y).shape[0]
    def get_nsamples(self):
        return self.x.shape[0]
class Encoder(torch.nn.Module):
    def __init__(self, channels):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding="same",)
        self.conv2 = torch.nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding="same",)
        self.conv3 = torch.nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding="same",)
        self.bn1 = torch.nn.BatchNorm2d(num_features = 24)
        self.bn2 = torch.nn.BatchNorm2d(num_features = 24)
        self.bn3 = torch.nn.BatchNorm2d(num_features = 32)
        self.rl = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.rl(x)

        return x


class Decoder(torch.nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.convt1 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1) ,padding=1,)
        self.convt2 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=24, kernel_size=(3, 3), stride=(1, 1) ,padding=1,)
        self.convt3 = torch.nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=(1, 1) ,padding=1,)
        self.conv4 = torch.nn.Conv2d(in_channels=24, out_channels=channels, kernel_size=(1, 1), stride=(1, 1), padding="same",)
        self.bn1 = torch.nn.BatchNorm2d(num_features = 32)
        self.bn2 = torch.nn.BatchNorm2d(num_features = 24)
        self.bn3 = torch.nn.BatchNorm2d(num_features = 24)
        self.rl = torch.nn.ReLU()
    def forward(self, x):
        x = self.convt1(x)
        x = self.bn1(x)
        x = self.rl(x)

        x = self.convt2(x)
        x = self.bn2(x)
        x = self.rl(x)

        x = self.convt3(x)
        x = self.bn3(x)
        x = self.rl(x)

        x = self.conv4(x)
        return x

class DOT(torch.nn.Module):
    def __init__(self, Encoder, Decoder, n):
        super(DOT, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        #self.mse = torch.nn.MSELoss(size_average=False)
        self.mse = torch.nn.MSELoss()
        self.initialize_weights()
        self.C = torch.nn.Parameter(5.0e-5 * torch.ones(n, n))

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
                    #torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
 
    def SelfExpress(self, h):
        h_temp = h.permute(0, 2, 3, 1)
        z = h_temp.flatten(1)
        n = z.shape[0]
        z_hat = torch.matmul(self.C, z)
        h_temp = z_hat.view(h_temp.shape)
        h_hat = h_temp.permute(0, 3, 1, 2)
        return z, z_hat, self.C, h_hat  

    def forward(self, x):
        h = self.Encoder(x)
        z, z_hat, C, h_hat = self.SelfExpress(h)
        x_hat = self.Decoder(h_hat)
        return z, z_hat, x_hat, C

    def loss_wass(self, z, z_hat, x, x_hat, C, REG_LATENT, WEIGHT_DECAY):
        #loss = self.mse(z, z_hat) + self.mse(x, x_hat)

        #--------------------------------------------------
        m, _, _, _ = x.shape
        a = torch.ones(m) / m
        x = x.reshape([m, -1]).cpu()
        x_hat = x_hat.reshape([m, -1]).cpu()
        M = ot.dist(x, x_hat)
        M = M / M.max()
        ls1 = ot.emd2(a, a, M)
        ls2 = self.mse(z_hat, z)
        ls3 = torch.norm(C, p = 2)

        #print(ls1.data.cpu().numpy(), ls2.data.cpu().numpy(), ls3.data.cpu().numpy())
        loss = ls1 + REG_LATENT * ls2 + WEIGHT_DECAY * ls3
        #--------------------------------------------------
        return loss
class SpectralClustering():
    def __int__(self):
        pass
    def predict(self, Coef, n_clusters, alpha=0.25):
        Coef = self.thrC(Coef, alpha)
        y_pre, C = self.post_proC(Coef, n_clusters, 8, 18)
        #np.savez('./models/Affinity.npz', coef=C)
        #np.savez('./models/y_pre.npz', y_pre=y_pre)
        # missrate_x = self.err_rate(y, y_x)
        # acc = 1 - missrate_x
        return y_pre, C

    def thrC(self, C, ro):
        if ro < 1:
            N = C.shape[1]
            Cp = np.zeros((N, N))
            S = np.abs(np.sort(-np.abs(C), axis=0))
            Ind = np.argsort(-np.abs(C), axis=0)
            for i in range(N):
                cL1 = np.sum(S[:, i]).astype(float)
                stop = False
                csum = 0
                t = 0
                while (stop == False):
                    csum = csum + S[t, i]
                    if csum > ro * cL1:
                        stop = True
                        Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                    t = t + 1
        else:
            Cp = C
        return Cp
    def build_aff(self, C):
        N = C.shape[0]
        Cabs = np.abs(C)
        ind = np.argsort(-Cabs, 0)
        for i in range(N):
            Cabs[:, i] = Cabs[:, i] / (Cabs[ind[0, i], i] + 1e-6)
        Cksym = Cabs + Cabs.T
        return Cksym

    def post_proC(self, C, K, d, alpha):
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        C = 0.5 * (C + C.T)
        r = d * K + 1
        U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** alpha)
        L = L / L.max()
        L = 0.5 * (L + L.T)
        spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                              assign_labels='discretize')
        spectral.fit(L)
        grp = spectral.fit_predict(L) + 1
        return grp, L
    def cluster_accuracy(self, y_true, y_pre):
        Label1 = np.unique(y_true)
        nClass1 = len(Label1)
        Label2 = np.unique(y_pre)
        nClass2 = len(Label2)
        nClass = np.maximum(nClass1, nClass2)
        G = np.zeros((nClass, nClass))
        for i in range(nClass1):
            ind_cla1 = y_true == Label1[i]
            ind_cla1 = ind_cla1.astype(float)
            for j in range(nClass2):
                ind_cla2 = y_pre == Label2[j]
                ind_cla2 = ind_cla2.astype(float)
                G[i, j] = np.sum(ind_cla2 * ind_cla1)
        m = Munkres()
        index = m.compute(-G.T)
        index = np.array(index)
        c = index[:, 1]
        y_best = np.zeros(y_pre.shape)
        for i in range(nClass2):
            y_best[y_pre == Label2[i]] = Label1[c[i]]

        # # calculate accuracy
        err_x = np.sum(y_true[:] != y_best[:])
        missrate = err_x.astype(float) / (y_true.shape[0])
        acc = 1. - missrate
        nmi = normalized_mutual_info_score(y_true, y_pre)
        kappa = cohen_kappa_score(y_true, y_best)
        ca = self.class_acc(y_true, y_best)
        return acc, nmi, kappa, ca

    def class_acc(self, y_true, y_pre):
        """
        calculate each classes's acc
        :param y_true:
        :param y_pre:
        :return:
        """
        ca = []
        for c in np.unique(y_true):
            y_c = y_true[np.nonzero(y_true == c)]  # find indices of each classes
            y_c_p = y_pre[np.nonzero(y_true == c)]
            acurracy = accuracy_score(y_c, y_c_p)
            ca.append(acurracy)
        ca = np.array(ca)
        return ca
