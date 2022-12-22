class f:
    def __init__(self):
        self.a = []
        self.b = []

    def add(self, x):
        self.a.append(x)
        self.b.append(x)

    def __str__(self):
        return f'{self.a}, {self.b}'


f = f()
f.add(1)
f.add(2)
print(str(f))
