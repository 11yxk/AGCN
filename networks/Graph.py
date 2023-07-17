import torch
import numpy as np

def ed(m, n):
    return np.sqrt(np.sum((m - n) ** 2))


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)

    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

# def normalize_digraph(A):
#     Dl = np.sum(A, 0)
#     num_node = A.shape[0]
#     Dn = np.zeros((num_node, num_node))
#     for i in range(num_node):
#         if Dl[i] > 0:
#             Dn[i, i] = Dl[i]**(-1)
#     AD = np.dot(A, Dn)
#     return AD

def get_adjacency_matrix(length = 14, threshold=3):

    A = np.zeros((length**2,length**2))
    X, Y = np.meshgrid(range(length), range(length))

    X = X.reshape(-1)
    Y = Y.reshape(-1)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            src=np.array([X[i],Y[i]])
            tgt=np.array([X[j],Y[j]])
            dis = ed(src,tgt)
            if dis > threshold:

                A[i, j] = 0
            else:

                A[i, j] = 1
    A=normalize_undigraph(A)

    return A



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    a = get_adjacency_matrix(5)
    print(a)

    plt.imshow(a, cmap='gray')
    plt.show()





