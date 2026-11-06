"""Escaped, human-readable local quality reports; no GUI or acquisition imports."""
from __future__ import annotations
import html
import json


def render_report_html(report, locale='zh-CN'):
    zh = locale == 'zh-CN'
    def text(cn, en):
        return cn if zh else en
    def esc(value):
        return html.escape(str(value if value is not None else '—'))
    def table(rows, headers=None):
        head = '<tr>' + ''.join('<th>' + esc(x) + '</th>' for x in headers) + '</tr>' if headers else ''
        return '<table>' + head + ''.join('<tr>' + ''.join('<td>' + esc(x) + '</td>' for x in row) + '</tr>' for row in rows) + '</table>'
    counts = report.get('trials', {}).get('counts', {})
    raw = report.get('raw_blocks', {})
    synthetic = report.get('data_origin') in ('synthetic_demo', 'legacy_simulated')
    origin = text('合成演示 / 非真实脑电', 'SYNTHETIC DEMO / NOT REAL EEG') if synthetic else text('所选历史记录；来源按原文件声明，不推断真实性', 'Selected historical recording; no authenticity inferred')
    parts = ['<!doctype html><html><head><meta charset="utf-8"><title>NeuroStation Quality Report</title>',
             '<style>body{font:14px sans-serif;color:#20313d;background:#fff;margin:24px}h1,h2{color:#087c88}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #dfe6e9;padding:8px;text-align:left}th{background:#edf6f6}p{line-height:1.6}.note{background:#fff2d9;padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style></head><body>',
             '<h1>' + text('会话质量报告', 'Session quality report') + '</h1>',
             '<p class="note"><b>' + origin + '</b><br>' + text('未知不等于合格。报告不重新评分、不推断注视目标，不证明硬件或个人模型有效。', 'Unknown is not a pass. This report does not rescore EEG, infer gaze, or validate hardware / personal models.') + '</p>',
             table([(text('会话', 'Session'), report.get('session_id')),
                    (text('数据来源目录', 'Source directory'), report.get('source_session')),
                    (text('原记录状态', 'Recorded status'), report.get('recording_status')),
                    (text('来源清单与内容复核通过', 'Source inventory and contents recheck passed'), report.get('source_unchanged'))]),
             '<h2>' + text('1. 原始记录完整性', '1. Raw recording integrity') + '</h2>']
    if raw.get('status') == 'legacy_no_block_journal':
        parts.append('<p>' + text('旧会话没有连续数据块日志：仍可读取元数据与事件，不能据此确认逐块完整性或复现在线处理。', 'Legacy session has no block journal. Metadata and events remain readable; block integrity and online replay are unavailable.') + '</p>')
    else:
        parts.append(table([(text('已核验样本', 'Verified samples'), raw.get('verified_samples')),
                            (text('完整封存', 'Complete seal'), raw.get('complete')),
                            (text('问题 / 未提交文件', 'Issues / uncommitted files'), json.dumps({k: raw.get(k) for k in ('issues', 'uncommitted_files', 'temporary_files')}, ensure_ascii=False))]))
    signal = report.get('signal_quality_updates', {})
    parts.append('<h2>' + text('2. 保存的信号质量更新', '2. Recorded signal-quality updates') + '</h2>')
    if signal.get('status') != 'observed':
        parts.append('<p>' + text('没有新版在线质量日志，无法评估。不会从旧的总体标签推断逐块质量。', 'No v2 online-quality journal: not assessable. Block quality is not inferred from historical summary labels.') + '</p>')
    else:
        states = signal.get('counts', {})
        parts.append(table([(text('全部质量更新（不是试次）', 'All quality updates (not trials)'), signal.get('all')),
                            (text('通过 / 异常 / 未知 / 无法读取', 'Pass / abnormal / unknown / unreadable'),
                             ' / '.join(str(states.get(k, 0)) for k in ('pass','fail','unknown','unassessable'))),
                            (text('质量日志与封存哈希一致', 'Quality journal matches seal hash'), signal.get('seal_evidence_match'))]))
        if signal.get('seal_evidence_match') is not True:
            parts.append('<p class="note">' + text('质量日志封存未核验或不匹配：以下仅是当前文件内容的统计，不能当作已验证的在线结果。', 'Quality seal unverified or mismatched: counts describe current file contents only, not verified online results.') + '</p>')
        abnormal = signal.get('abnormal_channel_updates', {})
        if abnormal:
            parts.append(table(list(abnormal.items()), [text('出现异常的通道', 'Abnormal channel'), text('异常更新次数', 'Abnormal updates')]))
        parts.append('<p>' + text('这些次数的分母是质量更新记录，不是试次、异常持续时间或模型准确率；未知阈值不会被改判为合格。', 'Denominators are quality updates, not trials, abnormal duration or model accuracy. Unvalidated thresholds remain unknown.') + '</p>')
    parts += ['<h2>' + text('3. 试次统计与分母', '3. Trials and denominators') + '</h2>',
              table([(text('已记录试次', 'Recorded trials'), counts.get('all', 0)),
                     (text('计划存在但没有事件', 'Planned without events'), counts.get('planned_without_events', 0)),
                     (text('质量合格', 'Quality eligible'), counts.get('quality_eligible', 0)),
                     (text('模型接受', 'Model accepted'), counts.get('model_accepted', 0)),
                     (text('拒绝', 'Rejected'), counts.get('rejected', 0)),
                     (text('无法评估', 'Unassessable'), counts.get('unassessable', 0))]),
              '<p>' + text('连续演示没有试次和模型决策，分母为 0 时显示“无法计算”，不显示 100% 或 0% 准确率。呈现目标不是注视证明；人工注视标签不是眼动追踪。', 'Continuous demos have no trials or model decisions. A zero denominator is not assessable, never 100% or 0% accuracy. Presented targets are not gaze evidence; manual gaze labels are not eye tracking.') + '</p>']
    names = {'acceptance': text('试次接受率', 'Trial acceptance'),
             'presented_target_id_quality_scored_accuracy': text('按呈现标签计算的质量合格且已评分准确率', 'Quality-scored accuracy vs presented labels'),
             'gaze_target_id_quality_scored_accuracy': text('按人工注视标签计算的质量合格且已评分准确率', 'Quality-scored accuracy vs manual gaze labels')}
    rates = []
    for key, value in report.get('trials', {}).get('rates', {}).items():
        n, d = value.get('numerator'), value.get('denominator')
        percent = text('无法计算', 'Not assessable') if not d or value.get('value') is None else f"{value['value'] * 100:.2f}%"
        rates.append((names.get(key, key), f'{n} / {d}', percent, value.get('population')))
    parts.append(table(rates, [text('指标', 'Metric'), text('分子 / 分母', 'Numerator / denominator'), text('结果', 'Result'), text('统计总体（原定义）', 'Population (original definition)')]))
    commands = report.get('commands', {})
    parts += ['<h2>' + text('4. 指令记录（不是准确率）', '4. Command records (not accuracy)') + '</h2>',
              table([(text('全部输出，含 UNKNOWN', 'All outputs, including UNKNOWN'), commands.get('all', 0)),
                     (text('接受', 'Accepted'), commands.get('accepted', 0)),
                     (text('拒绝或无法评估', 'Rejected / unassessable'), commands.get('rejected_or_unassessable', 0))]),
              '<h2>' + text('5. 逐试次原因', '5. Per-trial reasons') + '</h2>']
    reasons = {
        'aborted': text('已中止', 'Aborted'), 'scoring_failed': text('没有可用评分', 'No available score'),
        'planned_trial_without_events': text('计划试次没有事件', 'Planned trial without events'),
        'missing_or_ambiguous_events': text('缺少事件或事件不唯一', 'Missing / ambiguous events'),
        'unknown_sample_association': text('事件与样本关联未知', 'Unknown sample association'),
        'invalid_or_incomplete_interval': text('区间无效或不完整', 'Invalid / incomplete interval'),
        'score_without_events': text('只有评分而无事件', 'Score without events')}
    ledger = report.get('trial_ledger', [])
    # Keep desktop HTML bounded; the exported JSON always contains every row.
    rows = [(r.get('trial_index'), r.get('population'), r.get('decision'), reasons.get(r.get('reason'), r.get('reason')), r.get('presented_target_id'), r.get('gaze_target_id')) for r in ledger[:500]]
    parts.append(table(rows, [text('试次', 'Trial'), text('来源', 'Population'), text('决策', 'Decision'), text('原因', 'Reason'), text('呈现标签', 'Presented label'), text('注视标签', 'Gaze label')]) if rows else '<p>' + text('没有可列出的试次。', 'No trial rows.') + '</p>')
    if len(ledger) > 500:
        parts.append('<p>' + text('此视图只展示前 500 行；完整逐试次记录见 report.json。', 'Only the first 500 rows are displayed; report.json contains the full ledger.') + '</p>')
    parts += ['<h2>' + text('6. 来源哈希与限制', '6. Provenance and limitations') + '</h2>',
              table(list(report.get('source_hashes', {}).items())[:500], [text('文件', 'File'), 'SHA-256']),
              '<p>' + text('此视图至多展示 500 条来源哈希；完整清单保存在 report.json。', 'This view shows at most 500 source hashes; the full inventory is in report.json.') + '</p>',
              '<p>' + text('仅对已观察记录作软件检查；硬件阈值、物理刺激时序、临床及个人模型有效性均未由本报告验收。旧记录的历史结论未被改写。', 'Software checks cover observed records only. Hardware thresholds, physical stimulus timing, clinical validity and personal-model validity are not established. Historical conclusions are not rewritten.') + '</p>',
              '</body></html>']
    return ''.join(parts)
