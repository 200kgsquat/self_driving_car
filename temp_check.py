import sys, os  
sys.path.append('car-autopilot\\self-parking-ai-2d-with-pedestrians')  
from autopilot.parking import SmallParking  
p=SmallParking()  
inside=False  
for ped in p.pedestrians:  
    pos=ped.position  
    for rect in p.spaces.values():  
        x,y,w,h=rect  
