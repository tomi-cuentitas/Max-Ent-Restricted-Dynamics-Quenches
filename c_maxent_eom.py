#In [0]

import qutip
import numpy as np
import scipy.linalg


# In [1]:

def gram_matrix(basis: list, sp: Callable):
    """
    This module computes the Gram matrix associated to a basis of observables, for a quantum system, and a specific inner product.
    This module takes as input:
        *♥*♥* 1. basis: a list of quantum observables, in qutip.Qobj format.
        *♥*♥* 2. innerprod: the induced inner product, on the space of observables.
        *♥*♥* 3. (optional) as_qutip_qobj: boolean, default value: True.
                                            A boolean option which, if toggled on, returns the Gram matrix as a qutip.Qobj
        
        ====> Returns: the Gram matrix as an np.array.
        
        Warnings: (*). all entries must be square matrices. 
    
    """
    
    size = len(basis)
    result = np.zeros([size, size], dtype=float)

    for i, op1 in enumerate(basis):
        for j, op2 in enumerate(basis):
            if j < i:
                continue
            entry = np.real(sp(op1, op2))
            if i == j:
                result[i, i] = entry
            else:
                result[i, j] = result[j, i] = entry

    return result.round(14)

def orthogonalize_basis(basis: list, sp: Callable, idop=True):
    """
    This module orthonormalizes
        
        ====> Returns: a lambda function, given in terms of the state's eigenvalues and eigenvectors. 
    
    """
    
    if idop:
        idnorm_sq = sp(idop, idop)
        id_comp = [sp(idop, op) / idnorm_sq for op in basis]
        basis = ([idop * idnorm_sq**-0.5] +
                 [op - la for la, op in zip(id_comp, basis)])

    gs = gram_matrix(basis, sp)
    evals, evecs = np.linalg.eigh(gs)
    evecs = [vec / np.linalg.norm(vec) for vec in evecs.transpose()]
    return [
        p ** (-0.5) * sum(c * op for c, op in zip(w, basis))
        for p, w in zip(evals, evecs)
        if p > 0.00001
    ]

def project_op(op: Qobj, base_orthogonalisee: list, sp: Callable):
    return np.array([sp(qop, op) for qop in base_orthogonalisee])

# In [2]:

#def Hij_matrix(Hamiltonian: Qobj, base_orthogonalisee0

#def instantaneous_projs_and_avgs(basis: list, 
 #                                sp: Callable, 
  #                               e_ops: list):
   # 
    #basis_orth = orthogonalize_basis(basis = basis, 
     #                                sp = sp,
      #                               idop=True)
    #Kp_sp = sum(phia *opa for phia, opa in zip(=
