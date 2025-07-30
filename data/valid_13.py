
def fibonacci_generator():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

def get_fibonacci_sequence(n):
    fib = fibonacci_generator()
    return [next(fib) for _ in range(n)]
        