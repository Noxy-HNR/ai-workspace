"""Matching desktop symbols rendered from native geometry, with seven ICO sizes."""
from pathlib import Path
import struct
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
S=4
BG='12262F';INK='BBD4D9'
def color(h):
    r,g,b=(int(h[i:i+2],16) for i in (0,2,4));return b,g,r,255
def point(p):return tuple(round(v*S) for v in p)
def box(x,y,w,h,r,c):
    x,y,w,h,r=[round(v*S) for v in (x,y,w,h,r)];c=color(c)
    cv2.rectangle(canvas,(x+r,y),(x+w-r,y+h),c,-1)
    cv2.rectangle(canvas,(x,y+r),(x+w,y+h-r),c,-1)
    for px,py in ((x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)):cv2.circle(canvas,(px,py),r,c,-1,cv2.LINE_AA)
def circle(x,y,r,c,width=-1):cv2.circle(canvas,point((x,y)),round(r*S),color(c),round(width*S) if width>0 else -1,cv2.LINE_AA)
def line(points,c,width=7):
    pts=np.array([point(p) for p in points],np.int32)
    cv2.polylines(canvas,[pts],False,color(c),round(width*S),cv2.LINE_AA)
    for p in (points[0],points[-1]):circle(*p,width/2,c)
def polygon(points,c):cv2.fillPoly(canvas,[np.array([point(p) for p in points],np.int32)],color(c),cv2.LINE_AA)
def ellipse(x,y,a,b,angle,c,width=6):cv2.ellipse(canvas,point((x,y)),point((a,b)),angle,0,360,color(c),round(width*S),cv2.LINE_AA)
def base(accent):
    box(10,10,236,236,54,'14252F');box(17,17,222,222,48,'213B46');box(20,20,216,216,46,BG)
    box(88,210,80,5,2,accent)
def save(name):
    sizes=(16,24,32,48,64,128,256);layers=[];entries=[];offset=6+16*len(sizes)
    for size in sizes:
        ok,png=cv2.imencode('.png',cv2.resize(canvas,(size,size),interpolation=cv2.INTER_AREA))
        if not ok:raise RuntimeError('PNG encoding failed')
        blob=png.tobytes();layers.append(blob)
        entries.append(struct.pack('<BBBBHHII',size%256,size%256,0,0,1,32,len(blob),offset));offset+=len(blob)
    (ROOT/(name+'.ico')).write_bytes(struct.pack('<HHH',0,1,len(sizes))+b''.join(entries)+b''.join(layers))
    (ROOT/(name+'.png')).write_bytes(layers[-1])

projects=[('chemistry','Chemistry Workbench','78E6C6'),('research','Research Library','8DBFFF'),
          ('spectra','Spectroscopy Explorer','FFCC80'),('evaluation','AI Evaluation Lab','BEA4FF'),
          ('lecture','Lecture Notes','90DCE8'),('ai-stack','Local AI Stack','ADA8FF'),
          ('health','Personal Health','FF9FA6'),('ti84','TI-84 Chemistry','B8DD82')]
for name,label,accent in projects:
    canvas=np.zeros((256*S,256*S,4),np.uint8);base(accent)
    if name=='chemistry':
        polygon([(105,59),(151,59),(151,109),(192,178),(188,192),(68,192),(64,178),(105,109)],INK)
        polygon([(114,64),(142,64),(142,112),(181,180),(76,180),(114,112)],BG)
        polygon([(97,147),(156,138),(178,178),(78,178)],accent)
        line([(99,58),(157,58)],INK,8)
        circle(127,157,7,'D8FFF1');circle(150,168,4,'D8FFF1');circle(172,89,8,accent);circle(184,64,4,accent)
    elif name=='research':
        box(51,64,34,121,7,INK);box(57,72,22,105,3,'385764')
        box(92,56,37,129,7,accent);box(99,65,23,112,3,'284859')
        polygon([(142,73),(171,65),(201,178),(172,186)],INK)
        polygon([(151,78),(165,74),(191,173),(178,177)],'385764')
        line([(47,190),(207,190)],accent,7)
        line([(62,91),(74,91)],INK,4);line([(103,83),(117,83)],accent,4)
    elif name=='spectra':
        line([(53,62),(53,186),(207,186)],INK,6)
        for y in (94,131,163):line([(66,y),(203,y)],'29434C',2)
        line([(65,172),(85,172),(95,145),(105,171),(122,171),(137,77),(150,171),(164,171),(178,121),(191,171),(205,171)],accent,6)
        circle(137,77,8,accent);line([(137,55),(137,44)],accent,3)
    elif name=='evaluation':
        box(48,124,27,64,6,'554D79');box(88,98,27,90,6,'8176B2');box(128,140,27,48,6,accent)
        circle(174,91,37,accent);circle(174,91,29,'243943')
        line([(157,90),(169,102),(191,77)],accent,7)
        line([(48,193),(201,193)],INK,5)
    elif name=='lecture':
        box(65,51,133,145,13,INK);box(73,59,117,129,7,'1E3843')
        for y in (79,110,141,171):
            line([(54,y),(79,y)],accent,7)
        line([(100,81),(168,81)],accent,7)
        line([(100,106),(168,106)],INK,5);line([(100,130),(158,130)],INK,5)
        line([(100,155),(141,155)],INK,5)
    elif name=='ai-stack':
        for v in (89,115,141,167):
            line([(v,44),(v,66)],accent,6);line([(v,186),(v,202)],accent,6)
            line([(44,v),(66,v)],accent,6);line([(186,v),(207,v)],accent,6)
        box(65,65,122,122,20,INK);box(74,74,104,104,13,'203B49')
        for p,q in [((96,99),(151,99)),((96,99),(125,150)),((151,99),(125,150))]:line([p,q],accent,4)
        for x,y in ((96,99),(151,99),(125,150)):circle(x,y,11,accent);circle(x,y,4,'E6E0FF')
    elif name=='health':
        circle(95,103,39,accent);circle(161,103,39,accent)
        polygon([(58,115),(198,115),(128,188)],accent)
        line([(49,126),(89,126),(107,103),(126,154),(145,118),(158,126),(208,126)],BG,12)
        line([(49,126),(89,126),(107,103),(126,154),(145,118),(158,126),(208,126)],'E7FCF7',5)
    elif name=='ti84':
        box(72,45,112,154,17,INK);box(79,52,98,140,11,'29444D')
        box(89,65,78,41,6,accent);box(97,76,8,18,2,'3C5A42')
        line([(115,82),(148,82)],'3C5A42',4);line([(115,92),(138,92)],'3C5A42',4)
        for y in (124,146,168):
            for x in (99,128,157):box(x-9,y-7,18,14,4,accent if x==157 else INK)
    save(name)

# A labeled preview of the finished set, composited on a dark background.
sheet=np.full((530,960,3),(24,19,12),np.uint8)
for i,(name,label,accent) in enumerate(projects):
    icon=cv2.imread(str(ROOT/(name+'.png')),cv2.IMREAD_UNCHANGED)
    icon=cv2.resize(icon,(190,190),interpolation=cv2.INTER_AREA)
    x=(i%4)*240+25;y=(i//4)*265+18
    alpha=icon[:,:,3:4]/255
    sheet[y:y+190,x:x+190]=(icon[:,:,:3]*alpha+sheet[y:y+190,x:x+190]*(1-alpha)).astype(np.uint8)
    textsize=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.48,1)[0]
    cv2.putText(sheet,label,(i%4*240+(240-textsize[0])//2,y+220),cv2.FONT_HERSHEY_SIMPLEX,.48,(225,231,234),1,cv2.LINE_AA)
cv2.imwrite(str(ROOT/'project-icons-preview.png'),sheet)
print('Created 8 matching project icons, each with 7 resolutions.')
