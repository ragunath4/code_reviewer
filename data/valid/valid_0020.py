def odiiwf(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@odiiwf
def iyexii():
    return 'hello'
