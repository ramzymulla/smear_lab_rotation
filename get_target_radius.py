import math

global pixDistConv
global dur
global minTargetRadius
minTargetRadius = 150.0
dur = 1.0/30.0
@returns(tuple)
def process(value):
    # print(value[0])
    # print(value[1])
    da,dx,dy = value.Item1,value.Item2,value.Item3

    v = math.hypot(dx, dy)/dur # px/s
    angVel = abs(da)/dur # rad/s
    orbitalRadius = min(v/angVel if angVel > 0 else float(60000), 1000) # px
    # print(v)
    if v > 500 and 0:
        
        targetRadius = max(angVel, minTargetRadius)
        # print(angVel)
    else: 
        targetRadius = minTargetRadius
    return (targetRadius,angVel,v,orbitalRadius)
