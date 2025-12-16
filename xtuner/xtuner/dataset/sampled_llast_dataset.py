import random
from xtuner.dataset import LLaSTDataset

class SampledLLaSTDataset(LLaSTDataset):
    def __init__(self, *args, sample_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_data = self.data  # lưu toàn bộ dữ liệu gốc
        self.sample_size = sample_size
        self.resample()

    def resample(self):
        """Chọn ngẫu nhiên sample_size mẫu từ dữ liệu gốc"""
        if self.sample_size and self.sample_size < len(self.full_data):
            self.data = random.sample(self.full_data, self.sample_size)
        else:
            self.data = self.full_data

    def on_epoch_start(self):
        """Gọi lại khi bắt đầu epoch mới"""
        self.resample()
