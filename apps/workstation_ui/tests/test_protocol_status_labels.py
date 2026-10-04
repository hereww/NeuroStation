"""Display new formal protocols accurately while retaining legacy labels."""
from __future__ import annotations

import importlib.util
import unittest

from apps.workstation_ui.i18n import Translator


HAS_QT = importlib.util.find_spec("PySide6") is not None


@unittest.skipUnless(HAS_QT, "PySide6 not installed")
class ProtocolStatusLabelTests(unittest.TestCase):
    def test_formal_protocol_is_not_presented_as_a_candidate(self) -> None:
        from apps.workstation_ui.pages import DevicesPage

        self.assertEqual("formal", DevicesPage._protocol_status_key("formal"))
        self.assertEqual("formal", DevicesPage._protocol_status_key(" FORMAL "))
        self.assertEqual("正式", Translator("zh-CN")("device.protocol_status.formal"))
        self.assertEqual("Formal", Translator("en-US")("device.protocol_status.formal"))

    def test_old_candidate_and_draft_states_remain_readable(self) -> None:
        from apps.workstation_ui.pages import DevicesPage

        self.assertEqual("formal_candidate", DevicesPage._protocol_status_key("formal_candidate"))
        self.assertEqual("draft", DevicesPage._protocol_status_key("draft_for_workstation_mvp"))


if __name__ == "__main__":
    unittest.main()
