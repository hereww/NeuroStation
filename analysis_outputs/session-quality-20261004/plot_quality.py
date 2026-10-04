from pathlib import Path
import json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'comparison.json').read_text(encoding='utf-8'));r=sorted(r,key=lambda a:a['eye_side_file_label'])
W,H=1800,1500; im=Image.new('RGB',(W,H),'#f4f7fb');d=ImageDraw.Draw(im)
fontpath='C:/Windows/Fonts/msyh.ttc'
def font(size):return ImageFont.truetype(fontpath,size)
F=font(25);small=font(21);title=font(43);head=font(32)
C={'ink':'#173047','muted':'#53697b','grid':'#dbe4ec','blue':'#1c71c5','teal':'#168a84','gray':'#a8b6c5','white':'#ffffff'}
def text(pos,s,f=F,col=C['ink']):d.text(pos,str(s),font=f,fill=col)
def chart(rect,xlim,ylim,xticks,yticks,xlabel,ylabel):
 l,t,rr,b=rect
 for y in yticks:
  py=b-(y-ylim[0])/(ylim[1]-ylim[0])*(b-t);d.line((l,py,rr,py),fill=C['grid'],width=1)
  lab=f'{y:g}';box=d.textbbox((0,0),lab,font=small);text((l-12-(box[2]-box[0]),py-13),lab,small,C['muted'])
 for x in xticks:
  px=l+(x-xlim[0])/(xlim[1]-xlim[0])*(rr-l);d.line((px,t,px,b),fill=C['grid'],width=1)
  text((px-12,b+7),f'{x:g}',small,C['muted'])
 d.line((l,t,l,b,rr,b),fill=C['muted'],width=2)
 text((l,t-31),ylabel,small,C['muted']);text((rr-130,b+37),xlabel,small,C['muted'])
 def line(xs,ys,col,width=2):
  pts=[(l+(float(x)-xlim[0])/(xlim[1]-xlim[0])*(rr-l),b-(float(y)-ylim[0])/(ylim[1]-ylim[0])*(b-t)) for x,y in zip(xs,ys) if np.isfinite(y) and xlim[0]<=x<=xlim[1]]
  pts=[(px,min(b,max(t,py))) for px,py in pts]
  if len(pts)>1:d.line(pts,fill=col,width=width)
 return line
text((55,28),'NeuroStation · 两份采集的信号质量',title)
text((57,91),'按文件内左 / 右眼标记排列；显示的是采集质量，不是医学或眼功能评估。',F,C['muted'])
for col,res in enumerate(r):
 left=45+col*880;right=left+830
 d.rounded_rectangle((left,155,right,1385),radius=18,fill=C['white'])
 eye='左眼' if res['eye_side_file_label']=='left' else '右眼'
 text((left+26,175),eye+' · '+res['session_id'].split('session_')[1],head)
 text((left+28,223),'连续包序号 / 无已检出丢包 · 12 次试次完成',small,C['muted'])
 z=np.load(OUT/res['session_id']/'signals_for_plot.npz');raw=z['raw_uv'];clean=z['notched_band_uv'];freq=z['frequencies'];psd=z['psd_uv2_per_hz']
 text((left+26,274),'1  全通道慢漂移（各自减去起始基线）',F)
 rect=(left+95,348,right-25,555);plot=chart(rect,(0,100),(-50,10),[0,25,50,75,100],[-50,-30,-10,10],'时间 / 秒','变化 / mV')
 starts=np.arange(500,raw.shape[1]-500-249,250);meds=np.array([np.median(raw[:,j:j+250],axis=1) for j in starts]).T
 for i in range(8):plot(starts/250,(meds[i]-meds[i,0])/1000,C['gray'],2)
 plot(starts/250,(meds[6]-meds[6,0])/1000,C['blue'],3);plot(starts/250,(meds[7]-meds[7,0])/1000,C['teal'],3)
 text((left+100,610),'灰色 CH1–8   蓝色 CH7   青色 CH8',small,C['muted'])
 text((left+26,671),'2  后部四通道的原始功率谱',F)
 plot=chart((left+95,741,right-25,945),(1,65),(-1,5),[10,20,30,40,50,60],[-1,1,3,5],'频率 / Hz','log10(µV²/Hz)')
 plot(freq,np.log10(np.mean(psd[4:8],axis=0)+1e-12),C['blue'],3)
 frac=100*np.mean([c['line50_fraction_1_100'] for c in res['channels']]);text((left+100,995),f'全8通道50 Hz 占比均值：{frac:.1f}%（1–100 Hz）',small,C['muted'])
 text((left+26,1053),'3  滤波后的 CH7 / CH8 波形（12 Hz 试次）',F)
 tr=res['trials'][7];a=tr['start']+250;b=a+500;t=np.arange(500)/250
 plot=chart((left+95,1124,right-25,1320),(0,2),(-120,120),[0,.5,1,1.5,2],[-120,-60,0,60,120],'相对时间 / 秒','幅度 / µV')
 plot(t,clean[6,a:b],C['blue'],2);plot(t,clean[7,a:b],C['teal'],2)
text((55,1410),'滤波：50 / 100 Hz 陷波 + 1–40 Hz 零相位带通。滤波后仍可能保留眼动、肌电及电极伪迹。',small,C['muted'])
text((55,1446),'幅度按 BrainFlow Cyton 标准 µV 解释；通道位置来自自动配置，未独立核实。',small,C['muted'])
im.save(OUT/'quality-comparison.png')
print(OUT/'quality-comparison.png')
