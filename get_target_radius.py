import math

global pixDistConv
global dur
global minTargetRadius
minTargetRadius = 50.0
dur = 1.0/30.0
@returns(tuple)
def process(value):
    # print(value[0])
    # print(value[1])
    da,dx,dy = value.Item1,value.Item2,value.Item3

    v = math.hypot(dx, dy)/dur # cm/s
    angVel = abs(da)/dur # rad/s
    # print(v)
    if v > 500:
        
        targetRadius = max(angVel, minTargetRadius)
        # print(angVel)
    else: 
        targetRadius = minTargetRadius
    return (targetRadius,angVel,v)
