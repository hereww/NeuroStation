"""Operator-only quality panel. Metrics are not electrode/clinical validation."""
from collections import deque
import math
import time
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QTableWidget,QTableWidgetItem,QHeaderView


class QualityPlot(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent);self.setMinimumHeight(150);self.raw=[];self.display=[];self.psd=None

    def paintEvent(self,event):
        p=QPainter(self);p.fillRect(self.rect(),QColor('#111b29'));p.setRenderHint(QPainter.RenderHint.Antialiasing)
        width,height=self.width(),self.height(); p.setPen(QColor('#d3deef'))
        p.drawText(8,16,'原始 raw / 因果显示 display（µV；缺失保留断点）')
        wave_right=width*.67
        finite=[abs(v) for seq in (self.raw,self.display) for v in seq if v is not None and math.isfinite(v)]
        scale=max(max(finite,default=1),1)
        def draw(values,color,left,right,top,bottom,lo,hi):
            path=QPainterPath();active=False
            for i,v in enumerate(values):
                if v is None or not math.isfinite(v):active=False;continue
                x=left+i*(right-left)/max(1,len(values)-1);y=bottom-(v-lo)/(hi-lo)*(bottom-top)
                if active:path.lineTo(x,y)
                else:path.moveTo(x,y);active=True
            p.setPen(QPen(QColor(color),1));p.drawPath(path)
        draw(self.raw,'#97b7e8',8,wave_right,30,height-20,-scale,scale)
        draw(self.display,'#3ce5bb',8,wave_right,30,height-20,-scale,scale)
        p.setPen(QColor('#d3deef'));p.drawText(int(wave_right+10),16,'PSD · Welch 8s / 4s / 50%')
        if self.psd:
            logs=[10*math.log10(max(v,1e-20)) for v in self.psd]
            lo=min(logs)-1;hi=max(logs)+1
            draw(logs,'#f0b663',wave_right+10,width-8,30,height-20,lo,hi)
            p.setPen(QColor('#d3deef'));p.drawText(int(wave_right+10),height-3,'0                 125 Hz')
        else:
            p.drawText(int(wave_right+10),55,'等待完整频谱窗口')
        p.end()


class QualityPanel(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent);self.latest=None;self.identity=None;self.last_stop=-1
        self.raw=[];self.display=[]
        layout=QVBoxLayout(self);bar=QHBoxLayout();self.status=QLabel('质量：未知 / 等待数据');self.channel=QComboBox()
        bar.addWidget(self.status,1);bar.addWidget(self.channel);layout.addLayout(bar)
        self.table=QTableWidget(0,6);self.table.setHorizontalHeaderLabels(['通道','均值 µV','AC RMS µV','峰峰值 µV','漂移 µV','规则状态'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMaximumHeight(280);layout.addWidget(self.table)
        self.plot=QualityPlot();layout.addWidget(self.plot)
        self.details=QLabel('原始数据与模型专用处理互不替换。未知不等于合格。');self.details.setWordWrap(True);layout.addWidget(self.details)
        self.channel.currentIndexChanged.connect(self._plot)
        self.timer=QTimer(self);self.timer.setInterval(250);self.timer.timeout.connect(self._age);self.timer.start()

    def update_quality(self,result):
        if not isinstance(result,dict) or result.get('schema_version')!=2:return
        self.latest=result
        channels=result.get('channels',[])
        key=(result.get('stream_id'),result.get('processing_id'))
        if self.identity!=key:
            self.identity=key;self.last_stop=-1
            self.raw=[deque(maxlen=1250) for _ in channels];self.display=[deque(maxlen=1250) for _ in channels]
            self.channel.blockSignals(True);self.channel.clear();self.channel.addItems([r['channel'] for r in channels]);self.channel.blockSignals(False)
        if result.get('stop',-1)>self.last_stop:
            # File/UI polling may skip blocks. Never join non-adjacent samples as one trace.
            if self.last_stop >= 0 and result.get('start', self.last_stop) > self.last_stop:
                missing=min(1250,result['start']-self.last_stop)
                for buffer in (*self.raw,*self.display):buffer.extend([None]*missing)
            for buffers,name in ((self.raw,'raw_samples'),(self.display,'display_samples')):
                for b,values in zip(buffers,result.get(name,[])):b.extend(values)
            self.last_stop=result.get('stop',-1)
        self.table.setRowCount(len(channels))
        for i,row in enumerate(channels):
            values=[row['channel'],*[row.get(k) for k in ('mean_uv','ac_rms_uv','peak_to_peak_uv','drift_range_uv')],row.get('quality_status','unknown')]
            for j,v in enumerate(values):self.table.setItem(i,j,QTableWidgetItem('待评估' if v is None else f'{v:.2f}' if isinstance(v,float) else str(v)))
        intervals=result.get('artifacts',{}).get('intervals',[])
        text='；'.join(f"{r['channel']} {r['rule']} [{r['start']},{r['stop']})" for r in intervals[-6:]) or '当前块无已确认异常；不表示全部规则合格'
        self.details.setText(f"原因：{', '.join(result.get('reasons',[])) or '无'}\n{text}\n水位：{result.get('watermarks',{})} · 处理身份：{str(result.get('processing_id',''))[:12]}")
        self._age();self._plot()

    def _age(self):
        if not self.latest:return
        age=max(0,time.monotonic()-self.latest.get('updated_monotonic_s',0))
        state='过期 UNKNOWN' if age>2 else str(self.latest.get('dimensions',{}).get('signal_quality','unknown'))
        self.status.setText(f'质量：{state} · 更新距今 {age:.1f}s · 未经实机阈值验收')

    def _plot(self):
        i=self.channel.currentIndex()
        if i<0 or i>=len(self.raw):return
        self.plot.raw=list(self.raw[i]);self.plot.display=list(self.display[i])
        psd=(self.latest or {}).get('psd') or {};power=psd.get('power_uv2_hz',[])
        self.plot.psd=power[i] if i<len(power) else None;self.plot.update()
