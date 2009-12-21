import pymc as pm
import numpy as np

def adjust_scaled_val(index, delta, LI):
    """
    Returns L*(value-last_value), given that value and last_value
    differ only at index by an amount delta. LI is the inverse of L.
    """
    return LI[index]*delta
    
def logp_of_scaled(scaled_val, scaled_m):
    """
    Returns -.5 (val-m).T (L L.T).I (val-m)
    """
    return -.5*np.sum((scaled_val-scaled_m)**2)

class MVNMetropolis(pm.StepMethod):
    def __init__(self, stochastic, verbose=None, tally=True, proposal_sd=None):
        StepMethod.__init__(self, [stochastic, tally=tally])
        
        self._len = len(self.stochastic.value)
        
        self.adaptive_scale_factor = np.ones(self._len)
        self.accepted = np.zeros(self._len)
        self.rejected = np.zeros(self._len)
        self._state = ['rejected', 'accepted', 'adaptive_scale_factor']
        self._tuning_info = ['adaptive_scale_factor']
        
        if proposal_sd is None:
            proposal_sd = stochastic.value.copy()
            proposal_sd[np.where(proposal_sd==0)] = 1
        self.proposal_sd = proposal_sd
        
        self.stochastic = stochastic
        if verbose is not None:
            self.verbose = verbose
        else:
            self.verbose = stochastic.verbose
            
        self.L = self.stochastic.parents['L']
        self.mu = self.stochastic.parents['mu']
        self.LI = pm.Lambda('LI', lambda L=L: pm.gp.trisolve(L, np.eye(self._len)))
        
    def step(self):
        scaled_val = np.dot(self.LI.value, self.stochastic.value)
        scaled_m = np.dot(self.LI.value, pm.utils.value(self.mu))
        
        logp = logp_of_scaled(scaled_val, scaled_m)
        loglike = self.loglike
        
        proposal_sd = self.proposal_sd*self.adaptive_scale_factor
        
        for i in xrange(self._len):
            delta = np.random.normal(proposal_sd[i])
            new_val = self.stochastic.value.copy()
            new_val[i] += delta
            self.stochastic.value = new_val
            
            scaled_val_p = scaled_val + adjust_scaled_val(i, delta, self.LI.value)
            logp_p = logp_of_scaled(scaled_val_p, scaled_m)
            loglike_p = self.loglike
            
            if np.log(np.random.random())<logp_p+loglike_p-logp-loglike:
                scaled_val=scaled_val_p
                logp=logp_p
                loglike=loglike_p
            else:
                self.stochastic.revert()
        
    def tune(self):
        """
        Tunes the scaling parameter for the proposal distribution
        according to the acceptance rate of the last k proposals:

        Rate    Variance adaptation
        ----    -------------------
        <0.001        x 0.1
        <0.05         x 0.5
        <0.2          x 0.9
        >0.5          x 1.1
        >0.75         x 2
        >0.95         x 10
        
        """

        acc_rate_ = self.accepted / (self.accepted + self.rejected)
        tuning = 0*acc_rate

        for i in xrange(self._len):
            acc_rate = acc_rate_[i]
            # Switch statement
            if acc_rate<0.001:
                # reduce by 90 percent
                self.adaptive_scale_factor[i] *= 0.1
            elif acc_rate<0.05:           
                # reduce by 50 percent    
                self.adaptive_scale_factor[i] *= 0.5
            elif acc_rate<0.2:            
                # reduce by ten percent   
                self.adaptive_scale_factor[i] *= 0.9
            elif acc_rate>0.95:           
                # increase by factor of ten
                self.adaptive_scale_factor[i] *= 10.0
            elif acc_rate>0.75:           
                # increase by double      
                self.adaptive_scale_factor[i] *= 2.0
            elif acc_rate>0.5:            
                # increase by ten percent 
                self.adaptive_scale_factor[i] *= 1.1
            else:
                tuning[i] = False

        # More verbose feedback, if requested
        if verbose > 0:
            if hasattr(self, 'stochastic'):
                print '\t\tvalue:', self.stochastic.value
            print '\t\tacceptance rate:', acc_rate
            print '\t\tadaptive scale factor:', self.adaptive_scale_factor
            print

        # Re-initialize rejection count
        self.rejected *= 0.
        self.accepted *= 0.
        
        return np.any(tuning)