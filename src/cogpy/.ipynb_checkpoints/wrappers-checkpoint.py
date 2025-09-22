import numpy as np
from copy import deepcopy
from copy import copy
from tqdm import tqdm
import pandas as pd

def get_wrapper(attr_name):
    def _get_wrapper(method):
        def wrap_method(self, get=False, **kwargs):
            temp_self = copy(self)
            value = method(temp_self, **kwargs)
            if get:
                return value

            else:
                setattr(self, attr_name, value)
                
        return wrap_method
    return _get_wrapper

def dropnames(method):
    def wrap_method(self, *args, **kwargs):
        self.A = self.asarray()
        method(self, *args, **kwargs)
    return wrap_method

def flatten_arg(input, arg):
    output = [deepcopy(input) for i in range(len(input[arg]))]
    for i_elem, o_elem in zip(input[arg], output):
        o_elem[arg] = i_elem

    return output

def kwarg_intlist_wrapper(key_word):
    def _wrapper(func):
        """
        func: int -> object
        
        wrapped_func: list(int) -> list(object)
        """
        def wrapped_func(*args, **kwargs):
            if isinstance(kwargs[key_word], int):
                return [func(*args, **kwargs)]

            flat_kwarg = flatten_arg(kwargs, key_word)
            flat_output = np.empty_like(kwargs[key_word], object)
            for i, wrap_kwarg in enumerate(flat_kwarg):
                if isinstance(wrap_kwarg[key_word], int):
                    flat_output[i] = func(*args, **wrap_kwarg)
                
            return flat_output
        
        return wrapped_func

    return _wrapper


get_methods = lambda cls: [func for func in cls.__dict__ if callable(getattr(cls, func)) 
                           and not func.endswith('__')]
get_attrs = lambda cls: [_attr for _attr in cls.__dict__ if not _attr.startswith('__')]

def batch_method(func, _list):
    def _func(*args, progress_bar=True, **kwargs):
        wrap = (lambda x:x, tqdm)[progress_bar] # returns identity-function/tqdm if progressbar is False/True
        for elem in wrap(_list):
            func(elem, *args, **kwargs)
        
    return _func

# def batch_attr(_attr, _list):
    
#     def getter():
#         return [getattr(elem, _attr) for elem in _list]
    
#     def setter(value):
#         for elem in _list:
#             setattr(elem, _attr, value)
        
#     return property(getter, setter)
        

def batch_maker(cls):
    class Batch(cls):
        def __init__(self, batch, info=None):
            self._list = list(batch)
            if info is not None:
                self._info = list(info)

            else:
                self._info = [{'id':i} for i in range(len(self._list))]

            for method in get_methods(cls):
                if method in cls.__dict__:
                    setattr(self, method, batch_method(cls.__dict__[method], self._list))

            # for attr in get_attrs(self._list[0]):
            #     setattr(self, attr, [getattr(elem, attr) for elem in self._list])

        def __copy__(self):
            return Batch([copy(elem) for elem in self._list], copy(self._info))
        
        def __getitem__(self, i):
            return self._list[i]

        def __len__(self):
            return len(self._list)
        
        def __repr__(self):
            return f'Batch containing {len(self._list)} {cls.__name__} instances'

        def getattrib(self, attr):
            return [getattr(obj, attr) for obj in self._list]

        @property
        def _filt_log(self):
            return self.getattrib('_filt_log')

        @property
        def df(self, obj_name='sig'):
            return pd.DataFrame.from_dict([info|{obj_name: obj} for info, obj in zip(self._info, self._list)])

    return Batch

def add_docs_for(other_func):  
    def dec(func):  
        func.__doc__ = other_func.__doc__ + "\n\n" + func.__doc__
        return func
    return dec

from functools import partial

def partial_method(applicator, method):
    func = partial(applicator, method=method)
    func.__doc__ = applicator.__doc__ + \
        """

        Method
        ______

        """ + \
        method.__doc__
    return func
