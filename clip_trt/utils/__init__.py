#!/usr/bin/env python3
from .image import *
from .table import *
from .tensor import *


class AttributeDict(dict):
    """
    A dict where keys are available as attributes:
    
      https://stackoverflow.com/a/14620633
      
    So you can do things like:
    
      x = AttributeDict(a=1, b=2, c=3)
      x.d = x.c - x['b']
      x['e'] = 'abc'
      
    This is using the __getattr__ / __setattr__ implementation
    (as opposed to the more concise original commented out below)
    because of memory leaks encountered without it:
    
      https://bugs.python.org/issue1469629
      
    TODO - rename this to ConfigDict or NamedDict?
    """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, value):
        self.__dict__ = value

'''    
class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
'''

     
