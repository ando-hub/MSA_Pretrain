# coding:utf-8


def replace(d):
    d['b'] = 10
    return


a = {'a': 0, 'b': 1, 'c': 2}

print(a)

for k, v in a.items():
    if k == 'b':
        v = 5
print(a)

replace(a)
print(a)

