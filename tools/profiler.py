import time
def timeit(f):
    def timed(*args, **kw):
        tick = time.time()
        res = f(*args, **kw)
        tock = time.time()
        class_name = type(args[0]).__name__
        print("{} {} function time is: {:.5f}".format(class_name, f.__name__, tock - tick)) # ex. Preprocess call funtion time: xxx
        return res
    return timed


