import math


radiusTarget = 100.0
minTargetRadius = 100.0
dt = 1.0/30.0

@returns(tuple)
def process(value):
    global radiusTarget
    global minTargetRadius
    global dt
    
    # print(value[0])
    # print(value[1])
    da,dx,dy = value.Item1,value.Item2,value.Item3


    

    headingAngle = math.atan2(dy, dx)

    v = math.hypot(dx, dy)/dt # px/s
    angVel = abs(da)/dt # rad/s
    orbitalRadius = min(v/angVel if angVel > 0 else float(60000), 1000) # px
    # print(v)
    # if v > 500 and 0:
        
    #     targetRadius = max(headingAngle*100, minTargetRadius)
    #     # print(angVel)
    # else: 
    #     targetRadius = minTargetRadius

    thresh = 1
    metric = headingAngle
    if v > 300 and abs(metric)>thresh:
        if radiusTarget < 500:
            radiusTarget += 15*dt*(abs(metric)/thresh)
    else:
        # radiusTarget = max(minTargetRadius, radiusTarget - 10*dur*(1-(v/300)))
        radiusTarget = minTargetRadius
    return (radiusTarget,angVel,v,orbitalRadius,headingAngle)
