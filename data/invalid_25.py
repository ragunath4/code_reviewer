
def generator_function():
    for i in range(10):
        if i % 2 == 0:
            yield i * 2
        else:
            yield i * 3
        # Missing closing
        