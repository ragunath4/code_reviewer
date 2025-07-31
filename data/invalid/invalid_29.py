
def validate_input(data):
    if not isinstance(data, (list, tuple)):
        raise TypeError("Data must be list or tuple")
    
    for item in data:
        if not isinstance(item, (int, float)):
            raise ValueError("All items must be numbers")
    
    return True
        