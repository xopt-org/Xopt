import matlab.engine
def f():
    
    
    names = matlab.engine.find_matlab()
    eng_id = names[0]
    eng = matlab.engine.connect_matlab(eng_id)
    
    x1 = 1.
    x2 = 2.
    result = eng.testeval(x1,x2,nargout=2)
    
    print(result)
    
    
for i in range(10000):
    print(i)
    f()