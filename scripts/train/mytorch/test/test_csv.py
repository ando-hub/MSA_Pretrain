# coding:utf-8

import csv

a = {'hoge': 1, 'piyo': ['huga', 'foo']}
b = [a]

with open('test/test.csv', 'w') as fp:
    writer = csv.DictWriter(fp, fieldnames=a.keys())
    writer.writeheader()
    writer.writerows(b)
