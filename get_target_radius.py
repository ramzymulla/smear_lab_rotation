import math



minTargetRadius = 100.0
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

    threshrange = [2,6]
    metric = angVel
    
    if v > 100 and abs(metric)>threshrange[0] and abs(metric) < threshrange[1]:
        if radiusTarget < 500:
            radiusTarget += 100*dt
    else:
        # radiusTarget = max(minTargetRadius, radiusTarget - 10*dur*(1-(v/300)))
        radiusTarget = max(minTargetRadius, radiusTarget-50*dt)
    return (radiusTarget,angVel,v,orbitalRadius,headingAngle)
