import numpy as np

# for the system with alpha = 4 and beta = 0

A = np.array([
    [1,1,1,1,1],
    [-4,-3,-2,-1,0],
    [(4**2),(3**2),(2**2),(1**2),0],
    [(-4**3),(-3**3),(-2**3),(-1**3),0],
    [(-4**4),(-3**4),(-2**4),(-1**4),0]
])

b = np.array([0,0,2,0,0])

res = np.linalg.solve(A,b)
print(A)
print(res)

# for the system with alpha, beta = 2

B = np.array([
    [1,1,1,1,1],
    [-2,-1,0,1,2],
    [4,1,0,1,4],
    [-8,-1,0,1,8],
    [16,1,0,1,16]
])

c = np.array([0,0,2,0,0])

res2 = np.linalg.solve(B,c)
print(B)
print(res2)



