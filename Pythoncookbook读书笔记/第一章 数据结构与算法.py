# 1.1 将序列赋值给单独的变量

p = (4, 5)
x, y = p

print(x, y)

data = ['acme', 50, 90.1, (2021, 2, 16)]
name, shares, price,date = data

print(name, shares, price, date)

name, shares, price, (year, month, day) = data

print(name, shares, price, year, month, day)

# name, shares, price, year, month, day = data  # error
# print(name, shares, price, year, month, day)


s = 'hello'

a,b,c,d,e = s
print(a, b, c, d, e)

data = ['acme', 50, 90.1, (2021, 2, 16)]

_, shares, price, _ = data 
print(shares, price)

# 1.2 解压可迭代对象赋值给多个变量
import math
import random



def drop_first_last(grades):
    first, *middle, last = grades
    print(grades)
    return sum(middle)/len(middle)

print(drop_first_last([random.randrange(1, 20) for x in range(15)]))

record = ('Dave', 'deva@example.com', '733-555-123', '847-555-1221')

name, email, *phone_numbers = record

print(name, email, phone_numbers)

*trailing, current = [10, 8, 1, 7, 9, 5, 10, 3] 

print(trailing, current)


records = [
    ('foo', 1, 2),
    ('bar', 'hello'),
    ('foo', 3, 4),
]

def do_foo(x, y):
    print('foo: ', x, y)

def do_bar(s):
    print("bar: ", s)

for tag, *args in records:
    if tag == 'foo':
        do_foo(*args)
    elif tag == 'bar':
        do_bar(*args)

line = 'nobody:*:-2:-2:Unprivileged User:/var/empty:/usr/bin/false'

uname, *fields, homedir, sh = line.split(":")
print(uname, *fields, homedir, sh) 

record = ('ACME', 50, 123.45, (12, 18, 2012))

name, *_, (*_, year) = record  # 丢弃list

print(name, year)

items = [1, 10, 7, 4, 5, 9]

head, *tail = items

print(head, tail)

items = [1, 2, 3, 4, 5]
def sum(items):
    print(items)
    head, *tail = items
    return head + sum(tail) if tail else head

print(sum(items))


# 下面的代码在多行上面做简单的文本匹配， 并返回匹配所在行的上面N行

from collections import deque

def search(lines, pattern, history=5):
    previous_lines = deque(maxlen=history)

    for line in lines:
        if pattern in line:
            yield line, previous_lines
        previous_lines.append(line)

with open(r"./aoo.txt", mode='r', encoding='utf-8') as fp:
    for line, prelines in search(fp, 'python', history=2):
        for x in prelines:
            print(x, end='')
        print(line, end='')
        print("-"*20)

# 使用 deque(maxlen=N) 构造函数会新建一个固定大小的队列。
# 当新的元素加入并且这个队列已满的时候， 最老的元素会自动被移除掉。
# 如果不设置大小，是无限大的队列，可在两端实现插入删除操作。
dd = deque(maxlen=2)
dd.append(1)
dd.append(2)
dd.append(3)

print(dd)  # deque([2, 3], maxlen=2) 
dd.appendleft(4)
print(dd)  # deque([4, 2], ..)

dd.pop()
print(dd)  # 4
dd.popleft()
print(dd)  # 2

# 队列首尾插入删除时间
import time

a = []
b = deque()


start_ = time.time()
# [a.insert(0, x) for x in range(100000)]

end_ = time.time()

# print(end_ - start_)  # 2.155745267868042


start_ = time.time()
# [b.insert(0, x) for x in range(100000)]

end_ = time.time()

# print(end_ - start_)  # 0.015650510787963867

import heapq
nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]

print(heapq.nlargest(3, nums))  # 获取最大的3个元素 [42, 37, 23]
print(heapq.nsmallest(3, nums))

portfolio = [
    {'name': 'IBM', 'shares': 100, 'price': 91.1},
    {'name': 'AAPL', 'shares': 50, 'price': 543.22},
    {'name': 'FB', 'shares': 200, 'price': 21.09},
    {'name': 'HPQ', 'shares': 35, 'price': 31.75},
    {'name': 'YHOO', 'shares': 45, 'price': 16.35},
    {'name': 'ACME', 'shares': 75, 'price': 115.65}
]
cheap = heapq.nsmallest(3, portfolio, lambda s:s['price'])  # 根据price选出最小3个值
print(cheap)


# 1.5 实现一个优先级队列，并且返回优先级最高的队列

import heapq

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0
    
    def push(self, item, privoriry):
        heapq.heappush(self._queue, (-privoriry, self._index, item))
        self._index += 1
    
    def pop(self):
        return heapq.heappop(self._queue)[-1]


class Item:
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return "Item({!r})".format(self.name)

q = PriorityQueue()

q.push(Item('foo'), 1)
q.push(Item('bar'), 5)
q.push(Item('spam'), 4)
q.push(Item('grok'), 1)


print(q._queue)
