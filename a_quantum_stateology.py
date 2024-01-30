# In [1]:

import qutip
import numpy as np
import scipy.linalg as linalg

# In [2]:

def ev_checks(rho, check_positive_definite=True, tol=1e-3):
    if isinstance(rho,qutip.Qobj):
        pass
    else:
        try:
            rho=qutip.Qobj(rho)
    
    if check_positive_definite:
        try:
            rho=rho.full()
            np.linalg.cholesky(rho)
        except:
            return False
        return True
    else:
        ev_list = sorted(rho.eigenenergies())
        
    