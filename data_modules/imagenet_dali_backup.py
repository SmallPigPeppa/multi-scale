import pytorch_lightning as pl
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator



class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir):
        super().__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir + '/train')
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu",
                                         size=(224,224),
                                         interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=self.coin())
        return [output, labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir):
        super().__init__(batch_size, num_threads, device_id, seed=12 + device_id)

        self.input = ops.FileReader(file_root=data_dir + '/val')
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.resize = ops.Resize(device="gpu", resize_shorter=256)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.resize(images)
        output = self.cmnp(images)
        return [output, labels]

class DALIDataset(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_threads):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_threads = num_threads

    def setup(self, stage):
        self.device_id = self.trainer.local_rank
        self.train_pipe = HybridTrainPipe(batch_size=self.batch_size,
                                           num_threads=self.num_threads,
                                           device_id=self.device_id,
                                           data_dir=self.data_dir)
        self.val_pipe = HybridValPipe(batch_size=self.batch_size,
                                       num_threads=self.num_threads,
                                       device_id=self.device_id,
                                       data_dir=self.data_dir)
        self.train_pipe.build()
        self.val_pipe.build()

    def train_dataloader(self):
        return DALIGenericIterator(self.train_pipe, ['data', 'label'],
                               self.batch_size, fill_last_batch=False)

    def val_dataloader(self):
        return DALIGenericIterator(self.val_pipe, ['data', 'label'],
                                   self.batch_size, fill_last_batch=False)

# class DALIDataset(pl.LightningDataModule):
#     def __init__(self, data_dir, batch_size, num_threads, num_gpus):
#         super().__init__()
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.num_threads = num_threads
#         self.num_gpus = num_gpus
#
#     def setup(self, stage):
#         self.train_pipes = [HybridTrainPipe(batch_size=self.batch_size,
#                                              num_threads=self.num_threads,
#                                              device_id=i,
#                                              data_dir=self.data_dir)
#                             for i in range(self.num_gpus)]
#         self.val_pipes = [HybridValPipe(batch_size=self.batch_size,
#                                          num_threads=self.num_threads,
#                                          device_id=i,
#                                          data_dir=self.data_dir)
#                           for i in range(self.num_gpus)]
#         for pipe in self.train_pipes + self.val_pipes:
#             pipe.build()
#
#     def train_dataloader(self):
#         return [DALIGenericIterator(pipe, ['data', 'label'],
#                                     self.batch_size, fill_last_batch=False)
#                 for pipe in self.train_pipes]
#
#     def val_dataloader(self):
#         return [DALIGenericIterator(pipe, ['data', 'label'],
#                                     self.batch_size, fill_last_batch=False)
#                 for pipe in self.val_pipes]
