import math


radiusTarget = 100.0
minTargetRadius = 100.0
dur = 1.0/30.0

@returns(tuple)
def process(value):
    global radiusTarget
    global minTargetRadius
    global dur
    # print(value[0])
    # print(value[1])
    da,dx,dy = value.Item1,value.Item2,value.Item3

    headingAngle = math.atan2(dy, dx)

    v = math.hypot(dx, dy)/dur # px/s
    angVel = abs(da)/dur # rad/s
    orbitalRadius = min(v/angVel if angVel > 0 else float(60000), 1000) # px
    # print(v)
    # if v > 500 and 0:
        
    #     targetRadius = max(headingAngle*100, minTargetRadius)
    #     # print(angVel)
    # else: 
    #     targetRadius = minTargetRadius

    if v > 300:
        if radiusTarget < 500:
            radiusTarget += 10*dur*(v/300)
    else:
        radiusTarget = max(minTargetRadius, radiusTarget - 10*dur*(1-(v/300)))
    return (radiusTarget,angVel,v,orbitalRadius,headingAngle)
