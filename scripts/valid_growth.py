from math import sqrt, floor

def a (N): return 1 + N
def b (N): return 1 + N + binomial(N,2)
def c (N):
    return sum([binomial(N, i) for i in range(1, int(floor(sqrt(N))) + 1)])
def d (N): return 2**(floor(N/2.0))
def e (N): return 2**N

def binomial(n,k):
   accum = 1
   for m in range(1,k+1):
      accum = accum*(n-k+m)/m
   return accum

def ans():
    functions = [a, b, c, d]
    N_max = 1000
    for N in range(1, N_max + 1):
        for f in functions:
            if f(N) > e(N):
                functions.remove(f)
                print("Function {0} failed at N={1}".format(f, N))
                print("Expected {0}, got {1}".format(e(N), f(N)))
    print("Tested up to N =", N_max)

if __name__ == '__main__':
    ans()

