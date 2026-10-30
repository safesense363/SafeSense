import unittest

from features.dataset_config import (
    get_label_maps,
    load_activity_config,
    load_dataset_schema,
)


class TestDatasetConfig(unittest.TestCase):

    def test_activity_classes(self):
        config = load_activity_config()
        names = [item["name"] for item in config["classes"]]

        self.assertEqual(
            names,
            [
                "standing",
                "walking",
                "sitting",
                "running",
                "falling",
                "lying_down",
            ],
        )

    def test_sequence_contract(self):
        config = load_activity_config()

        self.assertEqual(config["sequence"]["shape"], [30, 17, 3])
        self.assertEqual(config["sequence"]["dtype"], "float32")

    def test_label_maps_are_consistent(self):
        name_to_id, id_to_name = get_label_maps()

        self.assertEqual(name_to_id["falling"], 4)
        self.assertEqual(id_to_name[4], "falling")
        self.assertEqual(len(name_to_id), 6)

    def test_dataset_schema(self):
        schema = load_dataset_schema()

        self.assertIn("sequence_id", schema["required_fields"])
        self.assertIn("subject_id", schema["required_fields"])
        self.assertIn("ntu_rgbd", schema["allowed_values"]["source"])
        self.assertIn("safesense", schema["allowed_values"]["source"])


if __name__ == "__main__":
    unittest.main()
