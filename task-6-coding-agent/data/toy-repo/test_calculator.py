from calculator import add, factorial

def test_add_positive_numbers():
    assert add(2, 3) == 5

def test_add_negative_number():
    assert add(-2, 3) == 1

def test_factorial():
    assert factorial(5) == 120
