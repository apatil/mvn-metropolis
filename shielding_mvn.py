import pymc as pm
import numpy as np

class ShieldingMVN(pm.MvNormalChol):
    """
    Like MvNormalChol, but variables that depend on x[i] will cache based on the value
    of x[i] alone, NOT all of x.
    """
    def __init__(self, *args, **kwds):
        pm.MvNormalChol.__init__(self, *args, **kwds)
        self.indices = np.array([pm.Index(self.__name__+'_%i'%i, self, i, trace=False) for i in xrange(len(self.value))])
        self.degenerates = np.array([pm.Degenerate(self.__name__+'_%i*'%i, ind, value=ind.value, trace=False) for i,ind in enumerate(self.indices)])
        
    def set_value(self, value, force=False):
        pm.MvNormalChol.set_value(self, value, force)
        if hasattr(self, 'degenerates'):
            if self.last_value is not None:
                for i in xrange(len(self.value)):
                    if self.value[i] != self.last_value[i]:
                        self[i].value = self.value[i]
            else:
                for i in xrange(len(self.value)):
                    self[i].value = self.value[i]
                
    value = property(pm.MvNormalChol.get_value, set_value)
    
    def revert(self):
        if hasattr(self, 'degenerates'):
            for i in xrange(len(self.value)):
                if self.value[i] != self.last_value[i]:
                    self[i].revert()
        pm.MvNormalChol.revert(self)
        
    def __getitem__(self, index):
        return self.degenerates[index]