from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

from neurostation_contract import UserProfile
from eeg_tools.workstation.users import UserRegistry


class UserRegistryTests(unittest.TestCase):
    def profile(self, user_id: str = "", name: str = "张三") -> UserProfile:
        return UserProfile(
            user_id=user_id,
            name=name,
            age=28,
            gender="male",
            medical_conditions=("none",),
        )

    def test_persistent_profiles_generate_ids_and_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Datasets"
            registry = UserRegistry(root)
            created = registry.add(self.profile())
            self.assertEqual("U0001", created.user_id)
            self.assertTrue((root / "Users" / "registry.json").is_file())
            reloaded = UserRegistry(root)
            self.assertEqual("张三", reloaded.get("U0001").name)

    def test_keep_data_delete_marks_session_orphaned_and_restore_relinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Datasets"
            root.mkdir()
            session = root / "session_1"
            session.mkdir()
            (session / "session.json").write_text(
                json.dumps({"session_id": "session_1", "user_id": "U0001", "user_name": "张三"}),
                encoding="utf-8",
            )
            registry = UserRegistry(root)
            registry.add(self.profile())
            registry.delete("U0001", move_data=False)
            value = json.loads((session / "session.json").read_text(encoding="utf-8"))
            self.assertEqual("orphaned", value["user_link_status"])
            registry.restore("U0001")
            value = json.loads((session / "session.json").read_text(encoding="utf-8"))
            self.assertEqual("active", value["user_link_status"])

    def test_move_data_to_trash_restore_and_purge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Datasets"
            root.mkdir()
            session = root / "session_2"
            session.mkdir()
            (session / "session.json").write_text(
                json.dumps({"session_id": "session_2", "user_id": "U0001"}),
                encoding="utf-8",
            )
            registry = UserRegistry(root)
            registry.add(self.profile())
            registry.delete("U0001", move_data=True)
            self.assertFalse(session.exists())
            self.assertTrue((root / "Trash" / "Datasets" / "U0001").exists())
            registry.restore("U0001")
            self.assertTrue(session.exists())
            registry.delete("U0001", move_data=True)
            registry.purge("U0001")
            self.assertFalse((root / "Trash" / "Datasets" / "U0001").exists())

    def test_profile_validation_rejects_duplicate_and_invalid_medical_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = UserRegistry(Path(directory))
            registry.add(self.profile())
            with self.assertRaisesRegex(ValueError, "validation.user_id_duplicate"):
                registry.add(self.profile("U0001", "李四"))
        with self.assertRaisesRegex(ValueError, "validation.user_medical_none"):
            replace(self.profile("U0099"), medical_conditions=("none", "diabetes")).validate()

    def test_corrupt_registry_is_backed_up_and_recreated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Datasets"
            users_dir = root / "Users"
            users_dir.mkdir(parents=True)
            registry_path = users_dir / "registry.json"
            registry_path.write_text("{not valid json", encoding="utf-8")
            registry = UserRegistry(root)
            self.assertEqual((), registry.users)
            self.assertTrue((users_dir / "registry.json.corrupt").is_file())
            self.assertTrue(json.loads(registry_path.read_text(encoding="utf-8")))

    def test_registry_writes_secure_store_and_public_index_without_medical_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Datasets"
            registry = UserRegistry(root)
            registry.add(replace(self.profile("U0001", "张三"), medical_conditions=("diabetes",)))
            secure = root / "Users" / "registry.secure"
            public = root / "Users" / "registry.json"
            self.assertTrue(secure.is_file())
            value = json.loads(public.read_text(encoding="utf-8"))
            text = public.read_text(encoding="utf-8")
            self.assertNotIn("medical_conditions", text)
            self.assertEqual(1, len(value["users"]))
            self.assertIn("U0001", {item["user_id"] for item in value["users"]})

    def test_public_snapshot_redacts_identity_and_medical_details(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Datasets"
            registry = UserRegistry(root)
            registry.add(replace(self.profile("U0001", "张三"), medical_conditions=("diabetes",)))
            output = registry.export_public_snapshot(root / "public" / "users.json")
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("U0001", value["users"][0]["user_id"])
            self.assertNotIn("name", value["users"][0])
            self.assertNotIn("medical_conditions", value["users"][0])
            self.assertTrue(value["users"][0]["screening_recorded"])

    def test_renaming_user_rewrites_session_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Datasets"
            session = root / "session_rename"
            session.mkdir(parents=True)
            (session / "session.json").write_text(
                json.dumps({"participant_id": "U0001", "user_id": "U0001", "user_name": "张三"}),
                encoding="utf-8",
            )
            registry = UserRegistry(root)
            registry.add(self.profile("U0001"))
            registry.update("U0001", self.profile("U0042", "李四"))
            value = json.loads((session / "session.json").read_text(encoding="utf-8"))
            self.assertEqual("U0042", value["participant_id"])
            self.assertEqual("U0042", value["user_id"])
            self.assertEqual("李四", value["user_name"])


if __name__ == "__main__":
    unittest.main()
