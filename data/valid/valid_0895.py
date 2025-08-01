def awdtoz(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@awdtoz
def rbigpc(x):
    return x
