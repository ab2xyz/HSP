class MyClass(object):
    def __init__(self):
        self.name = hex(id(self))
x = MyClass()
print(x.name)
