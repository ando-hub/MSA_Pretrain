# coding: utf-8

def _append(x, i):
    x.append(i)
    return

def _ret(*lists):
    return lists

if __name__ == '__main__':
    a = []
    for i in range(10):
        _append(a, i)
    
    print(a)

    aa = _ret(i, i)
    print(aa)
    
    a1, a2 = _ret(i, i)
    print(a1, a2)
