def rclwdm(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@rclwdm
def qngqhn():
    return 'hello'
