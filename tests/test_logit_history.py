import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from models.data_loader import IndexedDataset
from models.logit_history import load_history_batch, record_history_batch


class NumberedDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return torch.tensor(index), index % 3


class StableSampleIdTest(unittest.TestCase):
    def test_indices_stay_attached_to_samples_when_loader_is_shuffled(self):
        torch.manual_seed(7)
        loader = DataLoader(
            IndexedDataset(NumberedDataset()),
            batch_size=4,
            shuffle=True,
        )

        for values, _, sample_ids in loader:
            self.assertTrue(torch.equal(values, sample_ids))

    def test_history_is_looked_up_by_dataset_index_not_batch_position(self):
        history = {}
        record_history_batch(
            history,
            torch.tensor([8, 2, 11]),
            torch.tensor([[8.0], [2.0], [11.0]]),
        )

        logits, valid = load_history_batch(
            history,
            torch.tensor([11, 8, 2]),
            batch_size=3,
            num_classes=1,
            device=torch.device("cpu"),
        )

        self.assertTrue(valid.all())
        self.assertTrue(
            torch.equal(logits.squeeze(1), torch.tensor([11.0, 8.0, 2.0]))
        )

    def test_missing_history_is_marked_invalid(self):
        history = {}
        record_history_batch(history, [4], torch.tensor([[1.0, 2.0]]))

        _, valid = load_history_batch(
            history,
            [4, 9],
            batch_size=2,
            num_classes=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(valid.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
