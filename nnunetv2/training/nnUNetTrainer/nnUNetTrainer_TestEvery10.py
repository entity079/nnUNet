import subprocess
import os
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_TestEvery10(nnUNetTrainer):

    def on_epoch_end(self):
        # Let nnU-Net do ALL logging safely
        super().on_epoch_end()

        # Run test inference every 10 epochs
        if (self.current_epoch + 1) % 10 == 0:
            self.run_test_inference()

    def run_test_inference(self):
        dataset_name = self.dataset_name
        configuration = self.configuration
        fold = self.fold
        trainer = self.__class__.__name__

        output_dir = os.path.join(
            self.output_folder,
            f"test_epoch_{self.current_epoch + 1}"
        )

        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "nnUNetv2_predict",
            "-i", f"{os.environ['nnUNet_raw']}/Dataset{dataset_name}/imagesTs",
            "-o", output_dir,
            "-d", dataset_name,
            "-c", configuration,
            "-f", str(fold),
            "-tr", trainer,
            "--disable_tta"
        ]

        subprocess.run(cmd, check=True)
