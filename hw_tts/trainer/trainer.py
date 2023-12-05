import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_tts.base import BaseTrainer
# from hw_tts.logger.utils import plot_spectrogram_to_buf
from hw_tts.utils import inf_loop, MetricTracker
from hw_tts.utils import MelSpectrogram, MelSpectrogramConfig


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            generator,
            msd,
            mpd,
            criterion,
            metrics,
            optimizer_g,
            optimizer_d,
            config,
            device,
            dataloaders,
            lr_scheduler_g=None,
            lr_scheduler_d=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(generator, msd, mpd, criterion, metrics, optimizer_g, optimizer_d, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler_g = lr_scheduler_g
        self.lr_scheduler_d = lr_scheduler_d
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "discriminator_loss", "generator_loss", "mel_loss", "fm_loss", "generator grad norm", "msd grad norm", "mpd grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "discriminator_loss", "generator_loss", "mel_loss", "fm_loss", *[m.name for m in self.metrics], writer=self.writer
        )

        self.transform = MelSpectrogram(MelSpectrogramConfig())

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "audio"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self, model_type):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            if model_type == "generator":
                clip_grad_norm_(
                    self.generator.parameters(), self.config["trainer"]["grad_norm_clip"]
                )
            elif model_type == "discriminator":
                clip_grad_norm_(
                    self.msd.parameters(), self.config["trainer"]["grad_norm_clip"]
                )
                clip_grad_norm_(
                    self.mpd.parameters(), self.config["trainer"]["grad_norm_clip"]
                )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.generator.train()
        self.msd.train()
        self.mpd.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("generator grad norm", self.get_grad_norm("generator"))
            self.train_metrics.update("msd grad norm", self.get_grad_norm("msd"))
            self.train_metrics.update("mpd grad norm", self.get_grad_norm("mpd"))
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Generator Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["generator_loss"].item()
                    )
                )
                self.logger.debug(
                    "Train Epoch: {} {} Discriminator Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["discriminator_loss"].item()
                    )
                )
                self.logger.debug(
                    "Train Epoch: {} {} Mel Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["mel_loss"].item()
                    )
                )
                self.logger.debug(
                    "Train Epoch: {} {} Feature matching Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["fm_loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "generator learning rate", self.lr_scheduler_g.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "discriminator learning rate", self.lr_scheduler_d.get_last_lr()[0]
                )
                self.writer.add_audio("audio sample", batch["generated_audio"][0], 22050)
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        self.lr_scheduler_g.step()
        self.lr_scheduler_d.step()
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        # LOG 3 VALIDATION PREDICTIONS

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)


        generated_audios = self.generator(**batch)
        if type(generated_audios) is dict:
            generated_audios = generated_audios["generated_audios"]
        generated_mels = self.transform(generated_audios.squeeze(1))
        lns = batch["audio"].shape[-1] - batch["audio"].shape[-1] % 256
        batch["audio"] = batch["audio"][:,:,:lns]
        batch["generated_audio"] = generated_audios[:,:,:lns]

        batch["generated_spectrogram"] = generated_mels[:,:,:-1]
        if is_train:
            self.optimizer_d.zero_grad()

        xs_d, xs_real_d, fms_d, fms_real_d = self.mpd(**batch, detach=True)
        xs_s, xs_real_s, fms_s, fms_real_s = self.msd(**batch, detach=True)
        d_loss = self.criterion["discriminator"](xs_real_s, xs_s) + self.criterion["discriminator"](xs_real_d, xs_d)
        batch["discriminator_loss"] = d_loss
        if is_train:
            batch["discriminator_loss"].backward()
            self._clip_grad_norm("discriminator")
            self.optimizer_d.step()
        
        metrics.update("discriminator_loss", batch["discriminator_loss"].item())



        if is_train:
            self.optimizer_g.zero_grad()
        mel_loss = self.criterion["mel"](batch["spectrogram"], batch["generated_spectrogram"])
        batch["mel_loss"] = mel_loss

        xs_d, xs_real_d, fms_d, fms_real_d = self.mpd(**batch)
        xs_s, xs_real_s, fms_s, fms_real_s = self.msd(**batch)

        fm_loss = self.criterion["fm"](fms_real_d, fms_d) + self.criterion["fm"](fms_real_s, fms_s)
        batch["fm_loss"] = fm_loss
        g_loss = self.criterion["generator"](xs_d) + self.criterion["generator"](xs_s)
        batch["generator_loss"] = g_loss
        if is_train:
            loss = batch["mel_loss"] + batch["fm_loss"] + batch["generator_loss"]
            loss.backward()
            self._clip_grad_norm("generator")
            self.optimizer_g.step()
        metrics.update("mel_loss", batch["mel_loss"].item())
        metrics.update("fm_loss", batch["fm_loss"].item())
        metrics.update("generator_loss", batch["generator_loss"].item())


        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.generator.eval()
        self.msd.eval()
        self.mpd.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)

        # add histogram of model parameters to the tensorboard
        for name, p in self.generator.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


    @torch.no_grad()
    def get_grad_norm(self, model_type, norm_type=2):
        if model_type == "generator":
            parameters = self.generator.parameters()
        elif model_type == "msd":
            parameters = self.msd.parameters()
        elif model_type == "mpd":
            parameters = self.mpd.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
