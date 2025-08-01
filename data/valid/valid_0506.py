def zczvic(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@zczvic
def dhbkho():
    return 'hello'

from typing import List, Dict
