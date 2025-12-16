# File: xtuner/engine/hooks/resample_dataset_hook.py
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class ResampleDatasetHook(Hook):
    """Hook để resample dataset mỗi đầu epoch"""
    
    def __init__(self, log_info=True):
        super().__init__()
        self.log_info = log_info
    
    def before_train_epoch(self, runner):
        """Được gọi trước mỗi epoch"""
        dataset = runner.train_dataloader.dataset
        
        if hasattr(dataset, 'on_epoch_start'):
            dataset.on_epoch_start()
            
            if self.log_info:
                runner.logger.info(
                    f"✅ [Epoch {runner.epoch + 1}] Resampled dataset: "
                    f"{len(dataset)} samples"
                )