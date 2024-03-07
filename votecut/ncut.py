import torch
import torch.nn.functional as F


def ncut(features, tau=0.15, eps=1e-5, eig_vecs=1):
    """
    Compute the normalized cut eigenvectors and eigenvalues. Ths function uses pytorch to compute batched eigenvectors.
    :param features: batched features of shape (batch_size, num_nodes, feature_dim)
    :param tau: threshold for the adjacency matrix
    :param eps: small value to add to the adjacency matrix to avoid division by zero
    :param eig_vecs: number of eigenvectors to compute. If eig_vecs is 1, then only the second-smallest eigenvector is
     returned
    :return: eigenvectors and eigenvalues, both of shape (batch_size, num_nodes, eig_vecs).
    If the eigenvalue computation fails, then (None, None) is returned
    """
    features = F.normalize(features, p=2, dim=-1)
    A = torch.bmm(features, features.permute(0, 2, 1))
    A = A > tau
    A = torch.where(~A, eps, A.double())

    # A has shape (batch_size, num_nodes, num_nodes)
    batch_size, num_nodes, _ = A.size()

    # Create diagonal degree matrix D
    D = torch.sum(A, dim=2)
    D_diag = torch.diag_embed(D)
    D_over_sqrt = torch.diag_embed(torch.sqrt(1.0 / D))

    # Compute normalized Laplacian L = D^(-1/2) * (D - A) * D^(-1/2)
    L = torch.matmul(D_over_sqrt, torch.matmul(D_diag - A, D_over_sqrt))
    try:
        # Compute the eigenvectors and eigenvalues of L
        eigenvalues, eigenvectors = torch.linalg.eigh(L, UPLO='L')
    except:
        # if eigh fails then D is not positive definite, and we should return Nones
        print("eigh failed")
        return None, None
    eigenvalues, eigenvectors = eigenvalues[:, 1:eig_vecs+1], eigenvectors[:, :, 1:eig_vecs+1]

    return eigenvectors, eigenvalues
