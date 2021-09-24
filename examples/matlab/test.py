import matlab.engine
names = matlab.engine.find_matlab()
def f():
    
    
    eng_id = names[0]
    # eng = matlab.engine.connect_matlab(eng_id)
    eng = matlab.engine.start_matlab()
    x1 = 1.
    x2 = 2.
    result = eng.testeval(x1,x2,nargout=2)
    
    print(result)
    
    
for i in range(10000):
    print(i)
    f()
