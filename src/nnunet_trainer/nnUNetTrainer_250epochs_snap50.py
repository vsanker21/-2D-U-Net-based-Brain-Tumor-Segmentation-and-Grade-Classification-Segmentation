from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_250epochs_snap50(nnUNetTrainer):
    """250-epoch budget; keeps an extra snapshot after epoch 50."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.current_epoch == 50 and self.local_rank == 0:
            self.save_checkpoint(join(self.output_folder, "checkpoint_ep50.pth"))
