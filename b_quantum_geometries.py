# In [0]

import qutip
import numpy as np
import scipy.linalg as linalg

import b_spin_ops as su2

# In [1]:

def commutator(A, B):
    """
    Given two square matrices, operators, this module computes its commutator. 
    This module takes as input the following parameters:
        *♥*♥* 1. A: a complex-valued matrix,
        *♥*♥* 2. B: another complex valued matrix.
        ====> Returns: A*B - B*A
        
        Warnings: This module first checks the compatibility of the matrix dimensions. 
                  Qutip.Qobj formats needed for both matrices.
    """
    if A.dims[0][0] == B.dims[0][0]: 
        pass
    else:
        raise Exception("Incompatible Qobj dimensions")
    return A*B - B*A

def anticommutator(A, B):
    """
    Given two square matrices, operators, this module computes its anticommutator. 
    This module takes as input the following parameters:
        *♥*♥* 1. A: a complex-valued matrix,
        *♥*♥* 2. B: another complex valued matrix.
        ====> Returns: A*B + B*A
        
        Warnings: This module first checks the compatibility of the matrix dimensions. 
                  Qutip.Qobj formats needed for both matrices.
    """
    if A.dims[0][0] == B.dims[0][0]: 
        pass
    else:
        raise Exception("Incompatible Qobj dimensions")
    return A*B+B*A

# In [2]:
## HS geometry

def fetch_HS_inner_prod(): 
    """
    This module instantiates a lambda function, associated to the Hilbert Schmidt inner product.
                 
        ====> Returns: Tr (op1.dag() * op2)
              
              Warnings: (a). The inputs must all be QuTip.Qobj
                        (b). An exception will be raised if the input operators have non-compatible
                             dimensions.
    """
    return lambda op1, op2: (op1.dag() * op2).tr()

# In [3]:
## static and dynamical correlation product  

def fetch_corr_inner_prod(sigma):
    """
    This module instantiates a lambda function, associated to the (static or dynamical) correlation scalar product.
    This correlation scalar product is calculated for a specific sigma-state, belonging to the family of correlation scalar products. 
    This module takes as input:
        *♥*♥* 1. sigma: a referential density operator, which prescribes possibly non-equal statistical weights 
                         to different operators, in the algebra of observables of a quantum system.
             
        ====> Returns: Tr (sigma * anticommutator(op1.dag() * op2))
        
        Warnings: (a). The inputs must all be QuTip.Qobj
    """
    return lambda op1, op2: .5 * (sigma * (anticommutator(op1.dag(), op2) ).tr())

# In [4]: 

def safe_expm_and_normalize(K, override=True):
    n_eig=sum(K.dims[0])
    if n_eig <= 16: 
        e0 = max(np.real(K.eigenenergies()))
    else:
        e0 = max(np.real(K.eigenergies(sparse='True', sort='high', eigvals=n_eig)))
    sigma = (K-e0*qutip.tensor([qutip.qeye(2) for k in K.dims[0]]))
    sigma = sigma/sigma.tr()
    sigma_operateur_densite = qstates.is_density_op(K)
    if not override:
        assert sigma_operateur_densite, "Erreur: sigma n'est pas une opératuer de densité"
    return sigma, sigma_operateur_densite

def logM(rho, svd = True):
    if isinstance(rho, qutip.Qobj):
        qutip_form = True
        dims = rho.dims
    else:
        qutip_form = False        

    if svd:            
        if qutip_form:
            rho = rho.full()
        U, Sigma, Vdag = linalg.svd(rho, full_matrices = False)
        matrix_log = U @ np.diag(np.log(Sigma)) @ U.conj().transpose() 
    else: 
        if qutip_form:
            eigvals, eigvecs = rho.eigenstates()
            matrix_log = sum([np.log(vl)*vc*vc.dag() for vl, vc in zip(eigvals, eigvecs)]) 
        else:
            rho = rho.full()
            eigvals, eigvecs = linalg.eigh(rho)
            return evecs @ np.array([np.log(ev)*np.array(f) for ev, f in zip(evals, evecs.transpose().conj())])
    
    if qutip_form:
        matrix_log = qutip.Qobj(matrix_log, dims)
    return matrix_log

def rel_entropy(rho, sigma, svd = True):
    if svd:
        val = (rho*(logM(rho, True) - logM(sigma, True))).tr()
    else:
        assert ((ev_checks(rho) and ev_checks(sigma))), "Either rho or sigma have negative ev."
        val = (rho*(logM(rho, False)-logM(sigma, False))).tr()
        if (abs(val.imag - 0)>1.e-10):
            val = None
            raise Exception("Either rho or sigma have negative ev.")
    return val.real
                                  
# In [5]:
## Kubo geometry 

def fetch_kubo_int_inner_prod(sigma):
    """
    This module instantiates the (integrand) of a specific sigma-weighted inner product, belonging to the family of state-dependent KMB
    inner products, associated to the sigma state. This module takes as input:
        *♥*♥* 1. sigma: a qutip.Qobj, namely a quantum state. 
        
        ====> Returns: a lambda function, given in terms of the state's eigenvalues and eigenvectors. 
        
              Warnings: (a). sigma must be a qutip.Qobj
                        (b). sigma must be a valid quantum state. 
                        
    This module computes a KMB inner product function, associated to the sigma state, from its integral form. 
    This module takes as input:
        *♥*♥* 1. sigma: a qutip.Qobj, namely a quantum state. 
        
        ====> Returns: a lambda function, given in terms of the state's eigenvalues and eigenvectors.
    """  

    evals, evecs = sigma.eigenstates()

    def return_func(op1, op2):
        return 0.01 * sum(
            (
                np.conj((v2.dag() * op1 * v1).tr())
                * ((v2.dag() * op2 * v1).tr())
                * ((p1) ** (1.0 - tau))
                * ((p1) ** (tau))
            )
            for p1, v1 in zip(evals, evecs)
            for p2, v2 in zip(evals, evecs)
            for tau in np.linspace(0.0, 1.0, 100)
            if (p1 > 0.0 and p2 > 0.0)
        )

    return return_func

    return lambda op1, op2: 0.01 * sum(
        (
            np.conj((v2.dag() * op1 * v1).tr())
            * ((v2.dag() * op2 * v1).tr())
            * p1 ** (1 - tau)
            * p1 ** (tau)
        )
        for p1, v1 in zip(evals, evecs)
        for p2, v2 in zip(evals, evecs)
        for tau in np.linspace(0, 1, 100)
    )


def fetch_induced_distance(sp):
    def distance(op1, op2):
        dop = op1 - op2
        return np.sqrt(sp(dop, dop))

    return distance