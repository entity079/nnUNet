import subprocess
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_TestEvery10(nnUNetTrainer):

    def on_epoch_end(self):
        super().on_epoch_end()

        if (self.current_epoch + 1) % 10 == 0:
            self.logger.log("val_dice", float(val_dice), self.current_epoch)
            self.logger.log("val_loss", float(val_loss), self.current_epoch)
            self.run_test_inference()

    def run_test_inference(self):
        dataset_name = self.dataset_name
        configuration = self.configuration
        fold = self.fold
        trainer = self.__class__.__name__

        output_dir = (
            f"{self.output_folder}/test_epoch_{self.current_epoch + 1}"
        )

        cmd = [
            "nnUNetv2_predict",
            "-i", f"$nnUNet_raw/Dataset{dataset_name}/imagesTs",
            "-o", output_dir,
            "-d", dataset_name,
            "-c", configuration,
            "-f", str(fold),
            "-tr", trainer,
            "--disable_tta"
        ]

        subprocess.run(" ".join(cmd), shell=True, check=True)
