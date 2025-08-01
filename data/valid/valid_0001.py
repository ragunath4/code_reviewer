def wrpyif(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@wrpyif
def nsizbc():
    return 'hello'

class Qtitzu:
    def __init__(self, value):
        self.value = value
    
    def rkmuwt(self):
        return self.value * 2
