"""Painting primitive for a preflighted four-condition display surface.

No window creation, event loop, timers, acquisition, or implicit animation here.
Coordinates describe the whole selected display, not the old eye-half stimulus.
"""
from __future__ import annotations


def target_pixel_rect(width, height, parameters, frame):
    w, h = width * parameters['target_width'], height * parameters['target_height']
    if parameters['shape'] == 'circle':
        w = h = min(w, h)
    return (width * frame.x - w / 2, height * frame.y - h / 2, w, h)


def paint_condition_state(painter, width, height, parameters, state):
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QPen
    p = parameters
    def gray(level):
        return QColor.fromRgbF(level, level, level)
    viewport = QRectF(0, 0, width, height)
    painter.fillRect(viewport, gray(p['background_level']))
    frame = state.frame
    if frame is not None and frame.target_visible:
        rect = QRectF(*target_pixel_rect(width, height, p, frame))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gray(p['on_level'] if frame.lit else p['off_level']))
        if p['shape'] == 'circle':
            painter.drawEllipse(rect)
        else:
            painter.drawRect(rect)
    # Fixed fixation point remains throughout cue, baseline, stimulus and rest.
    painter.setPen(QPen(gray(1 if p['background_level'] < .5 else 0), 2))
    painter.drawLine(round(width/2-5), round(height/2), round(width/2+5), round(height/2))
    painter.drawLine(round(width/2), round(height/2-5), round(width/2), round(height/2+5))
    if state.message and state.message != '+':
        painter.setPen(gray(1 if p['background_level'] < .5 else 0))
        painter.setFont(QFont('Sans Serif', max(14, round(height/24))))
        painter.drawText(QRectF(0, height*.1, width, height*.2), Qt.AlignmentFlag.AlignCenter, state.message)
