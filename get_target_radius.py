import math



minTargetRadius = 50.0
radiusTarget = minTargetRadius
dt = 1.0/30.0

@returns(tuple)
def process(value):
    global radiusTarget
    global minTargetRadius
    global dt
    
    # print(value[0])
    # print(value[1])
    da,dx,dy,dt = value.Item1,value.Item2,value.Item3,float(value.Item4)/1000


    

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

    threshrange = [2,5]
    metric = angVel
    growthRate = 200.0
    maxRadius = 300.0
    shrinkRate = 100.0*(growthRate/300.0)
    vThresh = 200.0
    
    if v > vThresh and abs(metric)>threshrange[0]:
        if radiusTarget < maxRadius:
            radiusTarget += growthRate*dt
    else:
        # radiusTarget = max(minTargetRadius, radiusTarget - 10*dur*(1-(v/300)))
        radiusTarget = min(max(minTargetRadius, radiusTarget-shrinkRate*dt),maxRadius)
    return (radiusTarget,angVel,v,orbitalRadius,headingAngle)
