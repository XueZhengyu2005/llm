def add(a, b):
    """Return the sum of two numbers."""
    return a + b

def factorial(n):
    if n < 0:
        raise ValueError('n must be non-negative')
    result = 1
    for value in range(2, n + 1):
        result *= value
    return result
