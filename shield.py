import pymc as pm
import numpy as np

def shield(cls,name):
    """
    Returns a 'shielded' version of cls.
    
    If f = pm.MvNormal('f',np.zeros(2),np.eye(2)), and 
    y = pm.Normal('y', f[0], 1)
    
    then anytime f changes y has to recompute its log-probability,
    even if f[1] has changed but f[0] has not.
    
    If f = shield(pm.MvNormal)('f', np.zeros(2), np.eye(2)),
    this is not the case: if f[1] changes but f[0] does not,
    y will not need to recompute its log-probability.
    
    Shielded versions of stochastic subclasses have some minor 
    computational overhead, and should only be used when necessary.
    """
    class newcls(cls):

        def __init__(self, *args, **kwds):
            cls.__init__(self, *args, **kwds)
            self.indices = np.array([pm.Index(self.__name__+'_%i'%i, self, i, trace=False) for i in xrange(len(self.value))])
            self.degenerates = np.array([pm.Degenerate(self.__name__+'_%i*'%i, ind, value=ind.value, trace=False) for i,ind in enumerate(self.indices)])

        def set_value(self, value, force=False):
            cls.set_value(self, value, force)
            if hasattr(self, 'degenerates'):
                if self.last_value is not None:
                    for i in xrange(len(self.value)):
                        if self.value[i] != self.last_value[i]:
                            self[i].value = self.value[i]
                else:
                    for i in xrange(len(self.value)):
                        self[i].value = self.value[i]

        value = property(cls.get_value, set_value)

        def revert(self):
            if hasattr(self, 'degenerates'):
                for i in xrange(len(self.value)):
                    if self.value[i] != self.last_value[i]:
                        self[i].revert()
            cls.revert(self)

        def __getitem__(self, index):
            return self.degenerates[index]
    
    if name is None:
        name = 'Shielded'+cls.__name__
    newcls.__name__=name
    newcls.__doc__ = cls.__doc__
    
    return newcls