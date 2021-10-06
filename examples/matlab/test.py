import matlab.engine
def f(eng):
    x1 = 1.
    x2 = 2.
    result = eng.testeval(x1,x2,nargout=2)
    
names = matlab.engine.find_matlab()
eng_id = names[0]
eng = matlab.engine.connect_matlab(eng_id)
for i in range(10000):
    print(i)
    f(eng)