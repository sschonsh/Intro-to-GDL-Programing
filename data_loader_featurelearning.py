import numpy as np
import tensorflow as tf
from os import listdir
import scipy.io as spio


def cell_to_sparse(c, flag):
    # Unpack into tensor
    if flag == 1:
        ctemp = c.tocoo()
        ctemp = ctemp.transpose()
    else:
        ctemp = c.tocoo()
    indices = np.mat([ctemp.row, ctemp.col]).transpose()
    return tf.SparseTensorValue(indices, ctemp.data, ctemp.shape)

def m_to_tensor(w):
    wcoo = w.tocoo()
    indices = np.mat([wcoo.row, wcoo.col]).transpose()
    return tf.SparseTensorValue(indices, wcoo.data, wcoo.shape)
    
def get_stiff(address):
    mat = spio.loadmat(address, squeeze_me=True)
    surf   = mat['surf']
    SparseStiff = surf['S'][()]
    return cell_to_sparse(SparseStiff,0)
    
def get_dist(address):
    mat = spio.loadmat(address, squeeze_me=True)
    surf   = mat['surf']
    Dist = surf['Distances'][()]
    return Dist    


def get_mat(address, n_channel, level):
    multi_val = []
    multi_col = []
    multi_row = []

    mat = spio.loadmat(address, squeeze_me=True)
    surf   = mat['surf']
    m_cell = surf['MLM'][()]
    m_mat  = cell_to_sparse(m_cell[level],0)
    #m_mat  = m_to_tensor(m_val)
    h_mat = surf['pts'][()]
    if level < 3:
        interp_cell = surf['InterpUp'][()]
        interp_mat  = cell_to_sparse(interp_cell[level],0)
    else:
        interp_mat = []
    w_cell = surf['Wfull'][()]
    w_mat  = cell_to_sparse(w_cell[level],0)
    l_cell = surf['L'][()]
    l_mat  = l_cell[level]
    i_cell = surf['Ifull'][()]
    i_val  = np.asarray(i_cell[level]) - 1
    i_row_idx_cell = surf['IRowIndex'][()]
    i_row_idx = np.asarray(i_row_idx_cell[level]) - 1  # minus 1 for 0 indexing
    i_col_idx_cell = surf['IColumnIndex'][()]
    i_col_idx = np.asarray(i_col_idx_cell[level]) - 1

    n = 6890
    for j in range(n_channel):
        multi_val.extend(j * n + i_val)
        multi_col.extend(j * l_mat + i_col_idx)
        multi_row.extend(j * n + i_row_idx)

    val = np.int64(np.asarray(multi_val))
    r_array = np.int64(np.asarray(multi_row))
    c_array = np.int64(np.asarray(multi_col))
    idx     = np.transpose(np.mat([r_array, c_array]))
    dense_shape = np.int64(np.asarray([n * n_channel, len(multi_col)]))
    sample_cell  = surf['Sample'][()]
    sample = sample_cell[level]-1
    return m_mat, w_mat, h_mat, l_mat, val, idx, dense_shape, interp_mat, sample



def load_mat(path, n_channel, n_load = 0):
    files = [f for f in listdir(path) if f.endswith('.mat')]
    if n_load == 0:
        n_load = len(files)

    m_mat_1 = [None] * n_load
    m_mat_2 = [None] * n_load
    m_mat_3 = [None] * n_load
    m_mat_4 = [None] * n_load
    
    w_mat_1 = [None] * n_load
    w_mat_2 = [None] * n_load
    w_mat_3 = [None] * n_load
    w_mat_4 = [None] * n_load
    
    h_mat_1 = [None] * n_load
    h_mat_2 = [None] * n_load
    h_mat_3 = [None] * n_load
    h_mat_4 = [None] * n_load
    
    l_mat_1 = [None] * n_load
    l_mat_2 = [None] * n_load
    l_mat_3 = [None] * n_load
    l_mat_4 = [None] * n_load

    val_1 = [None] * n_load
    val_2 = [None] * n_load
    val_3 = [None] * n_load
    val_4 = [None] * n_load
    
    idx_1 = [None] * n_load
    idx_2 = [None] * n_load
    idx_3 = [None] * n_load
    idx_4 = [None] * n_load
    
    dense_shape_1  = [None] * n_load
    dense_shape_2  = [None] * n_load
    dense_shape_3  = [None] * n_load
    dense_shape_4  = [None] * n_load
    
    interp_mat_1 = [None] * n_load
    interp_mat_2 = [None] * n_load
    interp_mat_3 = [None] * n_load
    interp_mat_4 = [None] * n_load
    
    stiff = [None] * n_load
    dist  = [None] * n_load
    
 

    for i in range(n_load):
        m_mat_1[i], w_mat_1[i], h_mat_1[i], l_mat_1[i], val_1[i], idx_1[i], dense_shape_1[i], interp_mat_1[i], sample_1 = get_mat(path + files[i], n_channel, 0)
        m_mat_2[i], w_mat_2[i], h_mat_2[i], l_mat_2[i], val_2[i], idx_2[i], dense_shape_2[i], interp_mat_2[i], sample_2 = get_mat(path + files[i], n_channel, 1)
        m_mat_3[i], w_mat_3[i], h_mat_3[i], l_mat_3[i], val_3[i], idx_3[i], dense_shape_3[i], interp_mat_3[i], sample_3 = get_mat(path + files[i], n_channel, 2)
        m_mat_4[i], w_mat_4[i], h_mat_4[i], l_mat_4[i], val_4[i], idx_4[i], dense_shape_4[i], interp_mat_4[i], sample_4 = get_mat(path + files[i], n_channel, 3)
        stiff[i] = get_stiff(path + files[i])
        dist[i]  = get_dist(path + files[i])

        if i % 1 == 0:
            print('Loaded %2d files' % i)
    return stiff, dist, m_mat_1, w_mat_1, h_mat_1, l_mat_1, val_1, idx_1, dense_shape_1, interp_mat_1, m_mat_2, w_mat_2, h_mat_2, l_mat_2, val_2, idx_2, dense_shape_2, interp_mat_2, m_mat_3, w_mat_3, h_mat_3, l_mat_3, val_3, idx_3, dense_shape_3, interp_mat_3, m_mat_4, w_mat_4, h_mat_4, l_mat_4, val_4, idx_4, interp_mat_4, dense_shape_4
