import unittest

import torch

from models.batch_logit_history import compute_batch_dissimilarities
from models.model_cifar.resnet_GL import resnet32


class BatchLogitHistoryTest(unittest.TestCase):
    def setUp(self):
        self.model = resnet32(
            num_classes=2,
            num_branches=4,
            history_granularity='batch',
            dissimilarity_metric='wasserstein1',
            tau=1.0,
        )
        self.sample_ids = torch.tensor([10, 11])
        self.branch_logits = [
            torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
            torch.tensor([[0.0, 2.0], [0.0, 2.0]]),
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        ]
        self.branch_logits_stacked = torch.stack(self.branch_logits, dim=2)

    def assert_uniform(self, weights):
        for weight in weights:
            self.assertTrue(
                torch.allclose(weight, torch.full((2,), 1.0 / 3.0))
            )

    def test_first_two_epochs_are_uniform_then_next_epoch_uses_weights(self):
        epoch0_weights = self.model.compute_dissimilarities(
            self.branch_logits, self.sample_ids
        )
        self.assert_uniform(epoch0_weights)
        self.assertEqual(self.model.current_epoch_batch_distance_count, 0)

        epoch0_teacher = torch.zeros(2, 2)
        self.model.record_epoch_logits(
            self.sample_ids,
            epoch0_teacher,
            branch_logits=self.branch_logits_stacked,
        )
        self.model.update_epoch_history()

        epoch1_weights = self.model.compute_dissimilarities(
            self.branch_logits, self.sample_ids
        )
        self.assert_uniform(epoch1_weights)
        self.assertEqual(self.model.current_epoch_batch_distance_count, 0)

        branch_means = torch.stack(
            [logit.mean(dim=0) for logit in self.branch_logits]
        )
        expected_distances = compute_batch_dissimilarities(
            branch_means, torch.zeros(2), 'wasserstein1'
        )
        expected_weights = torch.softmax(expected_distances, dim=0)

        self.model.record_epoch_logits(
            self.sample_ids,
            sum(
                logit * weight.view(-1, 1)
                for logit, weight in zip(self.branch_logits, epoch1_weights)
            ),
            branch_logits=self.branch_logits_stacked,
        )
        self.assertEqual(self.model.current_epoch_batch_distance_count, 1)
        self.model.update_epoch_history()

        epoch2_weights = self.model.compute_dissimilarities(
            self.branch_logits, sample_ids=None
        )
        actual_weights = torch.stack([weight[0] for weight in epoch2_weights])
        self.assertTrue(torch.allclose(actual_weights, expected_weights))

    def test_batch_mode_does_not_populate_sample_history(self):
        self.model.record_epoch_logits(
            self.sample_ids,
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            branch_logits=self.branch_logits_stacked,
        )
        self.assertEqual(self.model.current_epoch_ensem_logits, {})
        self.assertEqual(self.model.prev_ensem_logits, {})
        self.assertTrue(
            torch.equal(
                self.model.prev_batch_teacher_mean,
                torch.tensor([0.5, 0.5]),
            )
        )

    def test_evaluation_does_not_accumulate_batch_distances(self):
        self.model.prev_batch_teacher_mean = torch.zeros(2)
        self.model.epoch_count = 1
        self.model.compute_dissimilarities(
            self.branch_logits, sample_ids=None
        )
        self.assertEqual(self.model.current_epoch_batch_distance_count, 0)


if __name__ == '__main__':
    unittest.main()
