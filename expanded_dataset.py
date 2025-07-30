import os
import random

def create_expanded_dataset():
    """Create a comprehensive dataset with diverse syntax errors and valid code"""
    
    # Valid code samples (complex structures)
    valid_samples = [
        # Basic structures
        ("valid_01.py", "def add(a, b):\n    return a + b"),
        ("valid_02.py", "for i in range(5):\n    print(i)"),
        ("valid_03.py", "if x > 0:\n    print('positive')\nelif x < 0:\n    print('negative')\nelse:\n    print('zero')"),
        ("valid_04.py", "class Calculator:\n    def __init__(self):\n        self.result = 0\n    \n    def add(self, x):\n        self.result += x\n        return self.result"),
        ("valid_05.py", "try:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print('Division by zero')\nfinally:\n    print('Cleanup')"),
        ("valid_06.py", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
        ("valid_07.py", "data = [x * 2 for x in range(10) if x % 2 == 0]"),
        ("valid_08.py", "with open('file.txt', 'r') as f:\n    content = f.read()"),
        ("valid_09.py", "def decorator(func):\n    def wrapper(*args, **kwargs):\n        print('Before')\n        result = func(*args, **kwargs)\n        print('After')\n        return result\n    return wrapper"),
        ("valid_10.py", "import numpy as np\n\ndef matrix_multiply(a, b):\n    return np.dot(a, b)"),
        
        # Complex nested structures
        ("valid_11.py", """
class DatabaseManager:
    def __init__(self, connection_string):
        self.connection = self._create_connection(connection_string)
    
    def _create_connection(self, conn_str):
        # Simulate connection creation
        return {"connected": True, "string": conn_str}
    
    def execute_query(self, query):
        if self.connection["connected"]:
            return {"status": "success", "data": []}
        else:
            raise ConnectionError("Not connected")
        """),
        
        ("valid_12.py", """
def process_data(data_list):
    results = []
    for item in data_list:
        if isinstance(item, dict):
            processed = {k: v * 2 for k, v in item.items() if isinstance(v, (int, float))}
            results.append(processed)
        elif isinstance(item, list):
            processed = [x * 2 for x in item if isinstance(x, (int, float))]
            results.append(processed)
    return results
        """),
        
        ("valid_13.py", """
def fibonacci_generator():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

def get_fibonacci_sequence(n):
    fib = fibonacci_generator()
    return [next(fib) for _ in range(n)]
        """),
        
        ("valid_14.py", """
class EventHandler:
    def __init__(self):
        self.listeners = {}
    
    def add_listener(self, event_type, callback):
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)
    
    def trigger_event(self, event_type, *args, **kwargs):
        if event_type in self.listeners:
            for callback in self.listeners[event_type]:
                callback(*args, **kwargs)
        """),
        
        ("valid_15.py", """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
        """),
        
        # Edge cases
        ("valid_16.py", ""),  # Empty file
        ("valid_17.py", "# This is a comment only"),
        ("valid_18.py", "import os\nimport sys\nimport json"),
        ("valid_19.py", "x = 1; y = 2; z = 3"),  # Multiple statements on one line
        ("valid_20.py", "def empty_function():\n    pass"),
    ]
    
    # Invalid code samples (diverse syntax errors)
    invalid_samples = [
        # Basic syntax errors
        ("invalid_01.py", "def test()\n    pass"),  # Missing colon
        ("invalid_02.py", "def test():\npass"),  # Indentation error
        ("invalid_03.py", "print('hello'"),  # Missing closing parenthesis
        ("invalid_04.py", "print('hello)"),  # Unclosed string
        ("invalid_05.py", "x = 5 6"),  # Missing operator
        ("invalid_06.py", "1name = 5"),  # Invalid variable name
        ("invalid_07.py", "data = [1, 2, 3"),  # Unclosed bracket
        ("invalid_08.py", "if True\n    print('yes')"),  # Missing colon
        ("invalid_09.py", "for i in range(5)\n    print(i)"),  # Missing colon
        ("invalid_10.py", "class Test\n    pass"),  # Missing colon
        
        # Complex syntax errors
        ("invalid_11.py", """
def complex_function(a, b, c
    if a > b:
        return a + b
    elif b > c:
        return b * c
    else:
        return a + b + c
        """),  # Missing closing parenthesis
        
        ("invalid_12.py", """
class MyClass:
    def __init__(self):
        self.data = []
    
    def add_item(self, item
        self.data.append(item)
        """),  # Missing closing parenthesis
        
        ("invalid_13.py", """
try:
    x = 1 / 0
except ZeroDivisionError
    print('error')
        """),  # Missing colon
        
        ("invalid_14.py", """
def nested_function():
    def inner():
        return 42
    return inner()
        """),  # Valid but complex structure
        
        ("invalid_15.py", """
def list_comprehension():
    return [x for x in range(10
        """),  # Unclosed bracket
        
        # Indentation errors
        ("invalid_16.py", """
def test_function():
    x = 10
  y = 20
    return x + y
        """),  # Inconsistent indentation
        
        ("invalid_17.py", """
if True:
    print('yes')
  print('wrong indentation')
        """),  # Wrong indentation
        
        # Missing elements
        ("invalid_18.py", """
def incomplete_function():
    if x > 0:
        return x
    # Missing else clause
        """),  # Valid but incomplete logic
        
        ("invalid_19.py", """
def missing_return():
    x = 10
    y = 20
    # Missing return statement
        """),  # Valid but incomplete
        
        ("invalid_20.py", """
def syntax_error_in_string():
    message = "This string is not closed
    return message
        """),  # Unclosed string
        
        # Advanced syntax errors
        ("invalid_21.py", """
def decorator_with_error(func):
    def wrapper(*args, **kwargs
        print('Before')
        result = func(*args, **kwargs)
        print('After')
        return result
    return wrapper
        """),  # Missing closing parenthesis
        
        ("invalid_22.py", """
class DatabaseError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message

class ConnectionError(DatabaseError):
    pass
        """),  # Missing closing parenthesis
        
        ("invalid_23.py", """
def matrix_operations():
    matrix = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9
    return matrix
        """),  # Unclosed bracket
        
        ("invalid_24.py", """
def complex_conditional():
    if x > 0 and y < 10 or z == 5:
        return True
    elif x < 0 and (y > 5 or z != 3:
        return False
    else:
        return None
        """),  # Missing closing parenthesis
        
        ("invalid_25.py", """
def generator_function():
    for i in range(10):
        if i % 2 == 0:
            yield i * 2
        else:
            yield i * 3
        # Missing closing
        """),  # Incomplete function
        
        # Real-world like errors
        ("invalid_26.py", """
import requests

def fetch_data(url):
    try:
        response = requests.get(url
        return response.json()
    except requests.RequestException as e:
        print(f"Error: {e}")
        return None
        """),  # Missing closing parenthesis
        
        ("invalid_27.py", """
def process_json_data(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = value.upper()
            elif isinstance(value, (int, float)):
                data[key] = value * 2
    return data
        """),  # Valid but complex
        
        ("invalid_28.py", """
class ConfigManager:
    def __init__(self, config_file):
        self.config = self.load_config(config_file
    
    def load_config(self, filename):
        with open(filename, 'r') as f:
            return json.load(f
        """),  # Missing closing parenthesis
        
        ("invalid_29.py", """
def validate_input(data):
    if not isinstance(data, (list, tuple)):
        raise TypeError("Data must be list or tuple")
    
    for item in data:
        if not isinstance(item, (int, float)):
            raise ValueError("All items must be numbers")
    
    return True
        """),  # Valid but complex validation
        
        ("invalid_30.py", """
def async_function():
    async def inner():
        await asyncio.sleep(1)
        return "done"
    
    return inner()
        """),  # Valid async structure
    ]
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Write valid samples
    for filename, code in valid_samples:
        with open(os.path.join('data', filename), 'w', encoding='utf-8') as f:
            f.write(code)
    
    # Write invalid samples
    for filename, code in invalid_samples:
        with open(os.path.join('data', filename), 'w', encoding='utf-8') as f:
            f.write(code)
    
    print(f"Created {len(valid_samples)} valid samples")
    print(f"Created {len(invalid_samples)} invalid samples")
    print(f"Total dataset size: {len(valid_samples) + len(invalid_samples)} samples")

if __name__ == '__main__':
    create_expanded_dataset() 