import copy
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import torch
import math
import scipy.sparse as sp
from scipy.sparse import linalg
from torch.nn.functional import normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Load_Dataset(Dataset):
    """ :param input shape: (num_hours, -1, num_features)
    """
    def __init__(self, feature_tensor, label_tensor):
        self.feature_tensor = feature_tensor
        self.label_tensor = label_tensor

    def __getitem__(self, idx):
        feature = self.feature_tensor[idx, :, :]
        label = self.label_tensor[idx, :, :]
        return feature, label

    def __len__(self):
        return self.feature_tensor.shape[0]

class Combine_Dataset(Dataset):
    def __init__(self, data_items, targets, data_relations):
        self.data_items = data_items
        self.targets = targets
        self.data_relations = data_relations

    def __getitem__(self, idx):
        data_items = self.data_items[idx, :, :, :]
        targets = self.targets[idx, :, :, :]
        data_relations = self.data_relations[idx, :, :]
        return data_items, targets, data_relations

    def __len__(self):
        return self.data_items.shape[0]


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def anom_detection_evaluation(y_hat, y_true):
    y_true = y_true.view(-1, 1).detach().numpy()
    acc = accuracy_score(y_true, y_hat)
    prec = precision_score(y_true, y_hat)
    recall = recall_score(y_true, y_hat)
    f1 = f1_score(y_true, y_hat)
    return acc, prec, recall, f1


def find_epsilon(errors, reg_level=1):
    """
    Threshold method proposed by Hundman et. al. (https://arxiv.org/abs/1802.04431)
    Code from TelemAnom (https://github.com/khundman/telemanom)
    """
    e_s = errors
    best_epsilon = None
    max_score = -10000000
    mean_e_s = np.mean(e_s)
    sd_e_s = np.std(e_s)

    for z in np.arange(2.5, 12, 0.5):
        epsilon = mean_e_s + sd_e_s * z
        pruned_e_s = e_s[e_s < epsilon]

        i_anom = np.argwhere(e_s >= epsilon).reshape(-1,)
        buffer = np.arange(1, 50)
        i_anom = np.sort(
            np.concatenate(
                (
                    i_anom,
                    np.array([i + buffer for i in i_anom]).flatten(),
                    np.array([i - buffer for i in i_anom]).flatten(),
                )
            )
        )
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        if len(i_anom) > 0:
            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s
            sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s
            if reg_level == 0:
                denom = 1
            elif reg_level == 1:
                denom = len(i_anom)
            elif reg_level == 2:
                denom = len(i_anom) ** 2

            score = (mean_perc_decrease + sd_perc_decrease) / denom

            if score >= max_score and len(i_anom) < (len(e_s) * 0.5):
                max_score = score
                best_epsilon = epsilon

    if best_epsilon is None:
        best_epsilon = np.max(e_s)
    return best_epsilon


def preprocess_adj_new(adj):
    adj_normalized = (torch.eye(adj.shape[0]).double() - (adj.transpose(0,1)))
    return adj_normalized


def preprocess_adj_new1(adj):
    adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double()-adj.transpose(0,1))
    return adj_normalized


def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr


def nll_gaussian(preds, target, variance, add_const=False):
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(torch.pow(mean1 - mean2, 2), 2.*np.exp(2. * variance))
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))


def kl_gaussian_sem(preds):
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0)))*0.5


# matrix loss: makes sure at least A connected to another parents for child
def A_connect_loss(A, tol, z):
    d = A.size()[0]
    loss = 0
    for i in range(d):
        loss +=  2 * tol - torch.sum(torch.abs(A[:,i])) - torch.sum(torch.abs(A[i,:])) + z * z
    return loss


# element loss: make sure each A_ij > 0
def A_positive_loss(A, z_positive):
    result = - A + z_positive * z_positive
    loss = torch.sum(result)

    return loss


def matrix_poly(matrix, d):
    x = torch.eye(d).double().to(device) + torch.div(matrix, d).to(device)
    return torch.matrix_power(x, d)


def h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A


def get_num_edges(seq):
    num_edges = 0
    for graph in seq:
        num_edges += len(np.nonzero(graph)[0])
    return num_edges


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def generate_sequential_data(feature_tensor, period):
    '''
    :param feature_tensor: (num_hours, num_counties, num_features)
    :param period: lag in Granger Causality
    :return: X_train, Y_train, num_training_hours = X_train.shape[0]
    '''

    num_hours = feature_tensor.shape[0]

    X_train = []
    Y_train = []

    for i in range(num_hours - 2*period):  # every two period, gets a pair of (X, Y) sample
        # print(i)
        # print(feature_tensor[i : i+period].shape)
        # print(feature_tensor[i + period: i + 2 * period].shape)
        X_train_periodically = feature_tensor[i: i+period]
        Y_train_periodically = feature_tensor[i+period: i+2*period]

        # print(X_train_periodically.shape)
        # print(Y_train_periodically.shape)

        X_train.append(X_train_periodically)
        Y_train.append(Y_train_periodically)

    X_train = torch.stack(X_train)
    Y_train = torch.stack(Y_train)

    return X_train, Y_train


def get_training_val_test_DYNDAG(DYNDAG, num_training_samples, num_val_samples, num_test_samples, num_counties, window_size):
    # DYNDAG.shape[0] -> num_days

    X_train_DYNDAG = np.zeros((num_training_samples+1, num_counties, num_counties))
    train_idx = 0
    for i in range(num_training_samples+1):
        train_idx = int(i / window_size)
        X_train_DYNDAG[i] = DYNDAG[train_idx]
    train_idx += 1  # skip one day, because Y_train occupies another window_size
    # print('last day in train X: {}'.format(train_idx))

    X_val_DYNDAG = np.zeros((num_val_samples+1, num_counties, num_counties))
    val_idx = 0
    for j in range(num_val_samples+1):
        val_idx = int(j / window_size) + train_idx
        X_val_DYNDAG[j] = DYNDAG[val_idx]
    val_idx += 1  # skip one day, because Y_val occupies another window_size
    # print('last day in val X: {}'.format(val_idx - train_idx))

    X_test_DYNDAG = np.zeros((num_test_samples+1, num_counties, num_counties))
    for k in range(num_test_samples+1):
        test_idx = int(k / window_size) + val_idx
        X_test_DYNDAG[k] = DYNDAG[test_idx]
    test_idx += 1  # skip one day, because Y_val occupies another window_size
    # print('last day in test X: {}'.format(test_idx - val_idx))

    return X_train_DYNDAG, X_val_DYNDAG, X_test_DYNDAG


def feature_l2_norm(feature):
    '''
    :param tensor: shape (num_counties, num_days, num_hours, num_features)
    :return:
    '''

    num_counties = feature.shape[0]
    num_days = feature.shape[1]
    num_hours = feature.shape[2]
    num_features = feature.shape[3]

    feature = feature.view(num_counties * num_days * num_hours, num_features)
    feature = normalize(feature, p=2.0, dim=0)
    feature = feature.view(num_counties, num_days, num_hours, num_features)

    return feature