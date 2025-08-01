def wfijlf():
    for i in range(10):
        yield i

def zavzie(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@zavzie
def dnejum():
    return 'hello'
