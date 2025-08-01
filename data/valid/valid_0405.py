def nlsdyb(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@nlsdyb
def xueuqe(x):
    return x
