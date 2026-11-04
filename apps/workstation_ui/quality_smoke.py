"""Full GUI workflow smoke; software fixtures only, never a hardware preview."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time


def run_quality_workflow_smoke(window, output):
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    from PySide6.QtGui import QFontDatabase, QFont
    font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc"
    if font.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0], 10))
    window.resize(1280, 1100)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    controller = window.quality_controller
    controller.output_root = output / 'jobs'
    controller.demo_options = {'seconds': 12, 'realtime': False}
    controller.replay_options = {'realtime_preview': False}
    checks = []
    started = time.monotonic()
    hardware_calls = []
    originals = {}

    def forbidden(*args, **kwargs):
        hardware_calls.append('unexpected_hardware_call')
        raise RuntimeError('Hardware/stimulus is forbidden during workflow verification')

    for name in ('start_ssvep', 'preflight_cyton', 'test_cyton_channel'):
        if hasattr(window.gateway, name):
            originals[name] = getattr(window.gateway, name)
            setattr(window.gateway, name, forbidden)

    def check(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})
        if not condition:
            raise AssertionError(name)

    def wait_idle():
        deadline = time.monotonic() + 35
        while controller.busy:
            app.processEvents()
            if time.monotonic() > deadline:
                raise TimeoutError('quality worker exceeded workflow deadline')
            time.sleep(.005)
        app.processEvents()
        check('job_' + str(controller.job), controller.phase == 'completed')

    error = None
    try:
        window.pages['apps'].quality_button.click()
        app.processEvents()
        page = window.pages['quality']
        check('main_navigation', window.current_page == 'quality')
        check('empty_state', page.panel.latest is None and not window.gateway.snapshot.active)
        page.demo_button.click()
        wait_idle()
        demo = controller.session
        check('synthetic_origin', json.loads((demo / 'session.json').read_text(encoding='utf-8'))['data_origin'] == 'synthetic_demo')
        check('all_channels_rendered', page.panel.table.rowCount() == 8)
        page.tabs.setCurrentIndex(0)
        window.grab().save(str(output / 'quality-workspace.png'))
        page.replay_button.click()
        wait_idle()
        verification = controller.verification
        check('verified_replay', verification['online_replay_match'] and verification['raw_hashes_unchanged'])
        check('sealed_demo', verification['recording_complete'] and verification['verified_samples'] == 3000)
        page.tabs.setCurrentIndex(1)
        page.report_button.click()
        wait_idle()
        check('report_visible', page.tabs.currentIndex() == 1 and bool(page.report_view.toPlainText()))
        report = controller.report['report']
        check('zero_denominator_unknown', report['trials']['rates']['acceptance']['value'] is None)
        check('source_unchanged', report['source_unchanged'])
        window.grab().save(str(output / 'quality-report.png'))
        controller.export_report(output / 'exports')
        wait_idle()
        check('report_export', (Path(controller.export_dir) / 'report.html').is_file())
        window.change_language('en-US')
        app.processEvents()
        check('english_view', window.tr('nav.quality') == 'Signal & quality' and window.pages['quality'].displayed_report is controller.report)
        window.change_language('zh-CN')
        check('no_hardware_calls', not hardware_calls and not window.gateway.snapshot.active)
        summary = {'source_session': str(demo), 'replay': verification,
                   'report_dir': controller.report['output_dir'], 'export_dir': controller.export_dir}
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        summary = {}
    finally:
        controller.cancel()
        deadline = time.monotonic() + 10
        while controller.busy and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        for name, original in originals.items():
            setattr(window.gateway, name, original)
        if not controller.busy:
            window.close()
        app.processEvents()
    result = {'schema_version': 1, 'checks': checks, 'all_passed': error is None and all(x['passed'] for x in checks),
              'error': error, 'elapsed_seconds': time.monotonic() - started,
              'hardware_calls': hardware_calls, 'software_fixtures_only': True, **summary}
    (output / 'workflow-verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result
