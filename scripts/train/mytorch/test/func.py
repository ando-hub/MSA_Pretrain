def test(a, b=1):
    return a + b*2

print(test(1))
print(test(1, 2))
print(test(1, b=2))
print(test(1, c=3))

