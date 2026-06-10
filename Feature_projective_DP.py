import os
import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from fvcore.nn import FlopCountAnalysis
from opacus import GradSampleModule
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.utils.data import DataLoader, Subset

from datautils.humanartsetblur import COCODataset
from TinyVit_mod import tiny_vit_5m_256_192
from utils.config_classification import cfg as cfg11
from utils.loss import AverageMeter, KLDiscretLoss
from utils.progress_bar import pit
from utils.transforms import flip_back_simdr, transform_preds


@dataclass
class TrainConfig:
    config_file: str = "INSERT_PATH_TO_HUMANART_CONFIG_YAML"
    dataset_root: str = "INSERT_PATH_TO_HUMANART_DATASET_ROOT"
    pretrained_checkpoint: str = "INSERT_PATH_TO_TINYVIT_PRETRAINED_CHECKPOINT"
    output_checkpoint: str = "INSERT_PATH_TO_OUTPUT_CHECKPOINT"
    eval_output_dir: str = "INSERT_PATH_TO_EVAL_OUTPUT_DIR"

    gpu: str = "cuda:0"
    seed: int = 42
    epochs: int = 25
    batch_size: int = 32
    num_workers: int = 8
    learning_rate: float = 1e-3
    simdr_split_ratio: float = 2.0

    epsilon: float = 0.2
    delta: float = 1 / 22000
    max_grad_norm: float = 1.0
    public_loss_weight: float = 0.1

    public_subset_size: int = 100
    projection_rank: int = 100
    projection_update_interval: int = 50

    flip_test: bool = True
    shift_heatmap: bool = True
    train_all_parameters: bool = False
    num_joints: int = 17
    image_width: int = 192
    image_height: int = 256


class ViTPose(nn.Module):
    def __init__(self, simdr_split_ratio: float = 2.0) -> None:
        super().__init__()
        self.simdr_split_ratio = simdr_split_ratio
        self.backbone = tiny_vit_5m_256_192(pretrained=False, num_classes=0)
        self.conn_head = nn.Conv2d(in_channels=320, out_channels=17, kernel_size=1)
        self.upsample = nn.Upsample(size=(32, 24), mode="bilinear", align_corners=True)
        self.mlp_head_x = nn.Linear(768, int(192 * simdr_split_ratio))
        self.mlp_head_y = nn.Linear(768, int(256 * simdr_split_ratio))

    def forward(self, x):
        features = self.backbone(x)[0]
        features = self.upsample(self.conn_head(features))
        features = features.view(features.shape[0], features.shape[1], -1)
        x_coord = self.mlp_head_x(features)
        y_coord = self.mlp_head_y(features)
        return x_coord, y_coord


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_partial_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        weights_only=False,
        map_location=torch.device("cpu"),
    )
    state_dict = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model.load_state_dict(filtered_state_dict, strict=False)


def set_trainable_parameters(model: nn.Module, train_all: bool) -> None:
    if train_all:
        for param in model.parameters():
            param.requires_grad = True
        return

    trainable_names = [
        "conn_head",
        "upsample",
        "mlp_head_x",
        "mlp_head_y",
        "backbone.layers.3",
    ]
    for name, param in model.named_parameters():
        param.requires_grad = any(layer in name for layer in trainable_names) or "norm" in name


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        param.numel()
        for param in model.parameters()
        if not trainable_only or param.requires_grad
    )


def strip_grad_sample_module_prefix(state_dict):
    return {
        key.replace("_module.", ""): value
        for key, value in state_dict.items()
    }


def mark_as_processed(obj) -> None:
    if isinstance(obj, torch.Tensor):
        obj._processed = True
    elif isinstance(obj, list):
        for item in obj:
            item._processed = True


def check_processed_flag_tensor(tensor: torch.Tensor) -> None:
    if hasattr(tensor, "_processed"):
        raise ValueError("Call zero_grad() before reusing per-sample gradients.")


def check_processed_flag(obj) -> None:
    if isinstance(obj, torch.Tensor):
        check_processed_flag_tensor(obj)
    elif isinstance(obj, list):
        for item in obj:
            check_processed_flag_tensor(item)


def generate_noise(
    std: float,
    reference: torch.Tensor,
    generator=None,
    secure_mode: bool = False,
) -> torch.Tensor:
    zeros = torch.zeros(reference.shape, device=reference.device)
    if std == 0:
        return zeros
    if secure_mode:
        torch.normal(mean=0, std=std, size=(1, 1), device=reference.device, generator=generator)
        noise_sum = zeros
        for _ in range(4):
            noise_sum += torch.normal(
                mean=0,
                std=std,
                size=reference.shape,
                device=reference.device,
                generator=generator,
            )
        return noise_sum / 2
    return torch.normal(
        mean=0,
        std=std,
        size=reference.shape,
        device=reference.device,
        generator=generator,
    )


def get_flat_grad_sample(param: torch.Tensor) -> torch.Tensor:
    if not hasattr(param, "grad_sample"):
        raise ValueError("Per-sample gradients were not found. Use GradSampleModule.")
    if param.grad_sample is None:
        raise ValueError("Per-sample gradients were not initialized.")
    if isinstance(param.grad_sample, torch.Tensor):
        return param.grad_sample
    if isinstance(param.grad_sample, list):
        return torch.cat(param.grad_sample, dim=0)
    raise ValueError(f"Unexpected grad_sample type: {type(param.grad_sample)}")


def get_epsilon(sigma: float, sampling_rate: float, steps: int, delta: float) -> float:
    accountant = create_accountant(mechanism="rdp")
    accountant.steps = [(sigma, sampling_rate, steps)]
    return accountant.get_epsilon(delta=delta)


def build_datasets(config: TrainConfig, normalize):
    cfg11.merge_from_file(config.config_file)

    train_dataset = COCODataset(
        cfg=cfg11,
        root=config.dataset_root,
        image_set="training",
        is_train=True,
        coord_representation="sa-simdr",
        simdr_split_ratio=config.simdr_split_ratio,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    val_dataset = COCODataset(
        cfg=cfg11,
        root=config.dataset_root,
        image_set="validation",
        is_train=False,
        coord_representation="sa-simdr",
        simdr_split_ratio=config.simdr_split_ratio,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    return train_dataset, val_dataset


def build_loaders(config: TrainConfig, train_dataset, val_dataset):
    public_count = min(config.public_subset_size, len(train_dataset))
    public_dataset = Subset(train_dataset, list(range(public_count)))
    public_loader = DataLoader(
        public_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
    )

    sampler = UniformWithReplacementSampler(
        num_samples=len(train_dataset),
        sample_rate=config.batch_size / len(train_dataset),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    return public_loader, train_loader, val_loader


def compute_projection_matrix(public_loader, model, criterion, device, rank: int) -> torch.Tensor:
    model.eval()
    grad_list = []

    for image, _blur_image, target_x, target_y, target_weight, _meta in public_loader:
        image = image.to(device)
        target_x = target_x.to(device)
        target_y = target_y.to(device)
        target_weight = target_weight.to(device)

        model.zero_grad()
        output_x, output_y = model(image)
        loss = criterion(output_x, output_y, target_x, target_y, target_weight)
        loss.backward()

        grad_parts = []
        for param in model.parameters():
            if param.requires_grad:
                grad_parts.append(param.grad.view(-1))
        grad_list.append(torch.cat(grad_parts).unsqueeze(0))

    gradients = torch.cat(grad_list, dim=0)
    gram = gradients @ gradients.t()
    eigvals, eigvecs = torch.linalg.eigh(gram)

    rank = min(rank, eigvals.numel())
    topk_indices = torch.argsort(eigvals, descending=True)[:rank]
    topk_eigvals = eigvals[topk_indices]
    topk_eigvecs = eigvecs[:, topk_indices]

    eps = 1e-8
    projection_basis = gradients.t() @ topk_eigvecs
    projection_basis = projection_basis / torch.sqrt(topk_eigvals.unsqueeze(0) + eps)
    return projection_basis


def flatten_model_gradients(model) -> torch.Tensor:
    grad_list = []
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            grad_list.append(param.grad.view(-1))
    if not grad_list:
        raise ValueError("No gradients were available to flatten.")
    return torch.cat(grad_list)


def assign_flattened_gradient(model, flat_grad: torch.Tensor) -> None:
    offset = 0
    for param in model.parameters():
        if param.requires_grad:
            numel = param.numel()
            param.grad.copy_(flat_grad[offset:offset + numel].view_as(param))
            offset += numel

    if offset != flat_grad.numel():
        raise ValueError("Flattened gradient size does not match trainable parameter size.")


def apply_private_gradients(model, config: TrainConfig, sigma: float, batch_size: int) -> None:
    grad_samples = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            check_processed_flag(param.grad_sample)
            grad_samples.append(get_flat_grad_sample(param))

    if not grad_samples:
        raise ValueError("No per-sample gradients were collected.")

    per_param_norms = [
        grad_sample.reshape(len(grad_sample), -1).norm(2, dim=-1)
        for grad_sample in grad_samples
    ]
    per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
    clip_factor = (config.max_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)

    for _, param in model.named_parameters():
        if param.requires_grad:
            if not hasattr(param, "summed_grad"):
                param.summed_grad = torch.zeros_like(param.data)
            grad_sample = get_flat_grad_sample(param)
            param.summed_grad += torch.einsum("i,i...", clip_factor, grad_sample)
            mark_as_processed(param.grad_sample)

    for param in model.parameters():
        if param.requires_grad:
            check_processed_flag(param.summed_grad)
            noise = generate_noise(
                std=sigma * config.max_grad_norm,
                reference=param.summed_grad,
            )
            param.grad = (param.summed_grad + noise).view_as(param) / batch_size
            mark_as_processed(param.summed_grad)

    for _, param in model.named_parameters():
        if param.requires_grad:
            param.grad_sample = None
            param.summed_grad = None


def project_private_gradients(model, projection_basis: torch.Tensor) -> None:
    flat_grad = flatten_model_gradients(model)
    projected_grad = projection_basis @ (projection_basis.t() @ flat_grad)
    assign_flattened_gradient(model, projected_grad)


def add_public_gradients(model, public_grads, weight: float) -> None:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    for param, public_grad in zip(trainable_params, public_grads):
        param.grad = param.grad + weight * public_grad


def evaluate(model, val_loader, val_dataset, criterion, config: TrainConfig, device):
    model.eval()
    idx = 0
    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.num_joints, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    losses = AverageMeter()

    with torch.no_grad():
        for image, _blur_image, target_x, target_y, target_weight, meta in pit(val_loader, color="red"):
            image = image.to(device)
            target_x = target_x.to(device)
            target_y = target_y.to(device)
            target_weight = target_weight.to(device)

            output_x, output_y = model(image)

            if config.flip_test:
                input_flipped = np.flip(image.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).to(device)
                output_x_flipped, output_y_flipped = model(input_flipped)

                output_x_flipped = flip_back_simdr(
                    output_x_flipped.cpu().numpy(),
                    val_dataset.flip_pairs,
                    type="x",
                )
                output_y_flipped = flip_back_simdr(
                    output_y_flipped.cpu().numpy(),
                    val_dataset.flip_pairs,
                    type="y",
                )
                output_x_flipped = torch.from_numpy(output_x_flipped.copy()).to(device)
                output_y_flipped = torch.from_numpy(output_y_flipped.copy()).to(device)

                if config.shift_heatmap:
                    output_x_flipped[:, :, :-1] = output_x_flipped.clone()[:, :, 1:]

                output_x = (output_x + output_x_flipped) * 0.5
                output_y = (output_y + output_y_flipped) * 0.5

            output_x = F.softmax(output_x, dim=2)
            output_y = F.softmax(output_y, dim=2)
            loss = criterion(output_x, output_y, target_x, target_y, target_weight)

            num_images = image.size(0)
            losses.update(loss.item(), num_images)

            center = meta["center"].numpy()
            scale = meta["scale"].numpy()
            score = meta["score"].numpy()

            max_val_x, preds_x = output_x.max(2, keepdim=True)
            max_val_y, preds_y = output_y.max(2, keepdim=True)
            mask = max_val_x > max_val_y
            max_val_x[mask] = max_val_y[mask]
            maxvals = max_val_x.cpu().numpy()

            output = torch.ones(
                [image.size(0), preds_x.size(1), 2],
                device=device,
            )
            output[:, :, 0] = torch.squeeze(preds_x / config.simdr_split_ratio)
            output[:, :, 1] = torch.squeeze(preds_y / config.simdr_split_ratio)
            output = output.cpu().numpy()

            preds = output.copy()
            for sample_idx in range(output.shape[0]):
                preds[sample_idx] = transform_preds(
                    output[sample_idx],
                    center[sample_idx],
                    scale[sample_idx],
                    [config.image_width, config.image_height],
                )

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = scale[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(scale * 200, axis=1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta["image"])
            idx += num_images

    name_values, perf_indicator = val_dataset.evaluate(
        all_preds,
        config.eval_output_dir,
        all_boxes,
        image_path,
    )
    return losses.avg, name_values, perf_indicator


def save_checkpoint(model, optimizer, epoch: int, perf_indicator: float, checkpoint_path: str) -> None:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    raw_model = model._module if hasattr(model, "_module") else model
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "model_state_dict": raw_model.state_dict(),
            "perf": perf_indicator,
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device(config.gpu if torch.cuda.is_available() else "cpu")

    model = ViTPose(simdr_split_ratio=config.simdr_split_ratio)
    load_partial_checkpoint(model, config.pretrained_checkpoint)
    set_trainable_parameters(model, train_all=config.train_all_parameters)

    flops = FlopCountAnalysis(
        model,
        torch.randn(1, 3, config.image_height, config.image_width),
    ).total()
    print(f"FLOPs: {flops}")
    print(f"Total parameters: {count_parameters(model)}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True)}")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_dataset, val_dataset = build_datasets(config, normalize)
    public_loader, train_loader, val_loader = build_loaders(config, train_dataset, val_dataset)

    model = model.to(device)
    criterion = KLDiscretLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20], 0.1)

    projection_model = deepcopy(model).to(device)
    projection_model.eval()

    model = GradSampleModule(model).to(device)

    sampling_rate = config.batch_size / len(train_dataset)
    sigma = get_noise_multiplier(
        target_epsilon=config.epsilon,
        target_delta=config.delta,
        sample_rate=sampling_rate,
        epochs=config.epochs,
    )
    print(f"Using sigma: {sigma}")

    best_ap = 0.0
    iteration = 0
    projection_basis = None

    for epoch in pit(range(config.epochs), color="green"):
        model.train()
        losses = AverageMeter()

        for image, blur_image, target_x, target_y, target_weight, _ in pit(train_loader, color="blue"):
            image = image.to(device)
            blur_image = blur_image.to(device)
            target_x = target_x.to(device)
            target_y = target_y.to(device)
            target_weight = target_weight.to(device)

            optimizer.zero_grad()

            output_x_public, output_y_public = model(blur_image)
            public_loss = criterion(
                output_x_public,
                output_y_public,
                target_x,
                target_y,
                target_weight,
            )
            trainable_params = [param for param in model.parameters() if param.requires_grad]
            public_grads = torch.autograd.grad(
                public_loss,
                trainable_params,
                retain_graph=True,
            )

            output_x_private, output_y_private = model(image)
            private_loss = criterion(
                output_x_private,
                output_y_private,
                target_x,
                target_y,
                target_weight,
            )
            private_loss.backward()

            if iteration % config.projection_update_interval == 0 or projection_basis is None:
                projection_model.load_state_dict(
                    strip_grad_sample_module_prefix(model.state_dict()),
                    strict=False,
                )
                projection_model.eval()
                projection_basis = compute_projection_matrix(
                    public_loader=public_loader,
                    model=projection_model,
                    criterion=criterion,
                    device=device,
                    rank=config.projection_rank,
                )

            apply_private_gradients(model, config, sigma, config.batch_size)
            project_private_gradients(model, projection_basis)
            add_public_gradients(model, public_grads, config.public_loss_weight)

            optimizer.step()
            iteration += 1

            losses.update(private_loss.item(), image.size(0))
            print(f"Loss: {losses.val:.3f} ({losses.avg:.3f})", end="\r")

        steps = (epoch + 1) * len(train_loader)
        cur_eps = get_epsilon(
            sigma=sigma,
            sampling_rate=sampling_rate,
            steps=steps,
            delta=config.delta,
        )
        print(f"\nEpoch {epoch} epsilon: {cur_eps:.3f}")
        print(f"Train loss: {losses.avg:.4f}")

        val_loss, name_values, perf_indicator = evaluate(
            model,
            val_loader,
            val_dataset,
            criterion,
            config,
            device,
        )
        print(f"Validation loss: {val_loss:.4f}")
        print(f"AP: {perf_indicator}")
        print(f"Per joint: {name_values}")

        if perf_indicator > best_ap:
            best_ap = perf_indicator
            save_checkpoint(model, optimizer, epoch, perf_indicator, config.output_checkpoint)
            print(f"Saved checkpoint for epoch {epoch}")

        lr_scheduler.step()
        print(f"Best AP so far: {best_ap}")


def main() -> None:
    config = TrainConfig()
    train(config)


if __name__ == "__main__":
    main()
