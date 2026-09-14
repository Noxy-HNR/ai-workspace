"""Build a multi-resolution Windows icon from simple camera geometry."""
from pathlib import Path
import struct
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
SCALE=4
canvas=np.zeros((256*SCALE,256*SCALE,4),dtype=np.uint8)

def color(value):
    value=value.lstrip('#');r,g,b=(int(value[i:i+2],16) for i in (0,2,4))
    return b,g,r,255

def box(x,y,w,h,r,fill):
    x,y,w,h,r=[round(v*SCALE) for v in (x,y,w,h,r)]
    c=color(fill)
    cv2.rectangle(canvas,(x+r,y),(x+w-r,y+h),c,-1)
    cv2.rectangle(canvas,(x,y+r),(x+w,y+h-r),c,-1)
    for cx,cy in ((x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)):
        cv2.circle(canvas,(cx,cy),r,c,-1,cv2.LINE_AA)

def circle(x,y,r,fill,width=-1):
    cv2.circle(canvas,(round(x*SCALE),round(y*SCALE)),round(r*SCALE),color(fill),round(width*SCALE) if width>0 else -1,cv2.LINE_AA)

# Compact camera silhouette with a prominent lens and separate IR emitters.
box(10,10,236,236,54,'14252F')
box(17,17,222,222,48,'213B46')
box(20,20,216,216,46,'12262F')
box(57,60,62,35,12,'A8C7CF')
box(34,79,188,117,25,'BBD4D9')
box(41,86,174,103,19,'28454F')
box(48,93,160,88,14,'162E38')
circle(115,136,46,'0B1B23')
circle(115,136,39,'78E6C6')
circle(115,136,31,'234C51')
circle(115,136,23,'0B202C')
circle(121,140,13,'245969')
circle(105,123,7,'C6FFEE')
circle(182,112,7,'FFB982')
circle(195,130,5,'FFB982')
circle(181,149,6,'FFB982')
box(88,210,80,5,2,'78E6C6')

sizes=(16,24,32,48,64,128,256)
layers=[]
for size in sizes:
    image=cv2.resize(canvas,(size,size),interpolation=cv2.INTER_AREA)
    ok,encoded=cv2.imencode('.png',image)
    if not ok:raise RuntimeError('Icon encoding failed')
    layers.append(encoded.tobytes())
offset=6+16*len(layers)
directory=[]
for size,png in zip(sizes,layers):
    directory.append(struct.pack('<BBBBHHII',size if size<256 else 0,size if size<256 else 0,0,0,1,32,len(png),offset))
    offset+=len(png)
(ROOT/'ir-camera.ico').write_bytes(struct.pack('<HHH',0,1,len(layers))+b''.join(directory)+b''.join(layers))
(ROOT/'ir-camera.png').write_bytes(layers[-1])
print('Created IR camera icon in seven sizes, 16–256 px.')
