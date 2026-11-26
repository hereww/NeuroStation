"""Human-readable companion to the per-decision and paired interval TSVs."""
from pathlib import Path
import sys
from statistics import median


def write_report(output, lock, summary, details, test, calibration):
    arms = summary['metrics']
    names = {'baseline': '免校准', 'personal': '个人校准'}
    paired = 'personal' in arms

    def table_row(cells):
        return '| ' + ' | '.join(cells) + ' |'

    def rate(n, d):
        return f'{n}/{d}（{n/d:.1%}）' if d else '无法评估'

    def diff(x, y):
        return f'{100*(y-x):+.1f}' if x is not None and y is not None else '—'

    def parameter(key):
        g = lock['gates'][key]
        return f"最高分 ≥ {g['min_score']:.2f}，领先第二名 ≥ {g['min_margin']:.2f}，连续 {g['confirmations']} 窗同一候选才报出"

    headings = [names[k] for k in arms]
    columns = ['指标', *headings] + (['个人减免校准（百分点）'] if paired else [])
    lines = ['# SSVEP 连续回放报告', '',
             f"回放场 `{summary['test']}`，共 {summary['sample_count']:,} 个样本。",
             f"个人校准来自较早场次 `{summary['calibration']}`。" if paired else '本次使用免校准参数。',
             '本报告评估从原始脑电重新计算的连续判定。实时队列中的延迟或丢弃可能影响现场显示。',
             '', table_row(columns), table_row(['---'] + ['---:'] * (len(columns) - 1))]
    for label, key, denom in [('闪烁首次报对','correct','stimulus_n'), ('闪烁错认','wrong','stimulus_n'),
                              ('闪烁漏检','miss','stimulus_n'), ('静息全段无目标','quiet','rest_n'),
                              ('静息段误报','false_alarm','rest_n'),
                              ('闪烁可评分覆盖','stimulus_n','stimulus_total'), ('静息可评分覆盖','rest_n','rest_total')]:
        values = [rate(m[key], m[denom]) for m in arms.values()]
        if paired:
            a, b = arms['baseline'], arms['personal']
            values.append(diff(a[key]/a[denom], b[key]/b[denom]) if a[denom] and b[denom] else '—')
        lines.append(table_row([label, *values]))
    if summary['exclusions'].get('headless_no_presentation'):
        lines.append('\n本场记录为无显示运行，软件时间线没有对应的屏幕呈现，闪烁及静息均无法评分。')
    lines.append('\n覆盖率分母包含场次记录中已完成或有事件、帧证据的试次。已结束的完整场次也核对计划总数。缺少成对标记的试次计入覆盖率分母并排除评分。')
    quality = summary['quality_windows']
    flat = {k:v for k,v in quality.items() if k.startswith('flat_channel_')}
    for reason, count in flat.items():
        channels = '、'.join('CH' + n for n in reason.removeprefix('flat_channel_').split('_'))
        lines.append(f'\n有 {count:,} 个判定窗内 {channels} 的全部读数保持不变。这些窗归为不可判定。')
    lines.extend(['', '## 各呈现频率的首次报对', '', table_row(['软件呈现', *headings]),
                  table_row(['---'] + ['---:'] * len(arms))])
    for hz in summary['frequencies']['baseline']:
        rows = [summary['frequencies'][arm][hz] for arm in arms]
        lines.append(table_row([f'{hz} Hz', *(rate(m['correct'], m['stimulus_n']) for m in rows)]))
    lines += ['', '## 首报与改对时间', '', '只计区间内的输出。按末样本归入左闭右开区间。首次目标错误后即使改对仍计错认。延时按样本差除以 250 换算。']
    for k in arms:
        name = names[k]
        first = [r[k+'_onset_to_first_target_s'] for r in details if r[k+'_result'] == 'correct']
        if first:
            lines.append(f'\n{name}首次报对的 {len(first)} 段，中位首报延时 {median(first):.2f} 秒。')
        recovered = [r[k+'_wrong_to_correct_s'] for r in details if r[k+'_wrong_to_correct_s'] is not None]
        text = f'，从首次错认到改对的中位时间为 {median(recovered):.2f} 秒' if recovered else ''
        lines.append(f"\n{name}错认 {summary['metrics'][k]['wrong']} 次，其中段内改对 {len(recovered)} 次{text}。")
    lines += ['', '## 参数与质量口径', '',
              f"检测方法 `{lock['method']}`。8 通道每 0.2 秒输出一次。短窗使用最近 2 秒脑电的典型相关系数（CCA）、邻频谱峰及最近 0.2 秒的相位一致性。实验事件只参与评分。",
              '', f"弱信号路径累积最近 {lock['evidence']['long_window']/lock['sampling_rate_hz']:.0f} 秒的相干能量，检验最近 1 秒和 0.2 秒是否仍支持该频率。综合分数用于排序，阈值和分差用于拒判。分数不代表概率。",
              '', '开机或断流恢复后，首个有效 2 秒窗中，两半都稳定存在的同频成分会作为背景扣除。若当时已经在闪烁，持续目标也可能漏检。运行中新增或相位变化的同频干扰仍可能误报。质量正常时这些结果照常参与评分。',
              '', '免校准采用预设工程阈值。'+parameter('baseline')+'。']
    if paired:
        lines += ['', '个人校准仅用较早场选择拒判参数。'+parameter('personal')+'。在 42 组候选中最大化（首次报对率 + 静息无误报率）/ 2，平分时优先报对率，再按固定网格顺序选择。校准场如缺静息则仅优化报对率。']
    lines += [
              '', '用原始样本序号、256 回绕的包号、EEG 有限性与时间戳检查连续性。时间戳不递增或相邻间隔超过 250 毫秒标为中断。更短的正向时间戳跳变单独不足以认定丢包。平直通道、断流后的未满窗及预热均标为不可用。',
              '', f"全场相邻时间戳超过 6 毫秒的次数为 {summary['time_steps_over_6ms']:,}。样本级异常计数（同一样本记优先原因）：`{summary['raw_issues']}`。",
              '', '闪烁按呈现目标的成对标记评分。仅将有熄灭结束帧、休息协议、可对齐的原始边界标记且无额外刺激标记的相邻试次间隔纳入静息。任何一个预期输出缺失或窗口质量不可用时，整段排除。完整、有效但均被模型拒判的闪烁段计漏检。没有可评分静息段时静息误报率无法评估。',
              '', '详细结果见 [逐试次与静息段](intervals.tsv)，包含排除原因、首次目标及样本、首次正确样本和两种延时。全场输出见 [免校准](baseline_predictions.tsv)' + (' 与 [个人校准](personal_predictions.tsv)' if paired else '') + '。频率列留空时，`state=no_target` 表示主动拒判，`unavailable` 表示未能判定。',
              '', '参数与源码指纹见 [冻结参数](frozen_models.json)。' + ('校准场全部候选见 [校准网格](calibration_grid.json)。' if paired else '') + '原始文件指纹及计数见 [机器可读结果](metrics.json)。',
              '',  '标签是软件呈现频率，实际注视和屏幕物理闪烁时刻未经独立测量。', '', '## 在本机重跑', '', '```powershell']
    def quote(path):
        return "'" + str(path).replace("'", "''") + "'"
    if getattr(sys, 'frozen', False) or hasattr(sys.modules['__main__'], '__compiled__'):
        command = '& ' + quote(sys.executable) + ' --ssvep-replay'
    else:
        command = '& ' + quote(sys.executable) + ' ' + quote(Path(__file__).resolve().parents[1] / 'analyze_ssvep_session.py')
    command += ' ' + quote(test.resolve()) + ' --continuous'
    if paired:
        command += ' --calibration-session ' + quote(calibration.resolve())
    lines.extend([command + " --output-dir (Join-Path $env:TEMP ('ssvep-' + [guid]::NewGuid()))", '```'])
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
