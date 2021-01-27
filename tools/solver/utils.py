
from functools import partial

import torch
import torch.nn as nn

from tools.solver.fastai_optim import OptimWrapper

from tools.solver import learning_schedules_fastai as lsf

def flatten_model(m):
    return sum(map(flatten_model, m.children()), []) if len(list(m.children())) else [m]

def get_layer_groups(m):
    return [nn.Sequential(*flatten_model(m))]

def build_one_cycle_optimizer(model, optimizer_config):
    if optimizer_config.fixed_wd:
        optimizer_func = partial(
            torch.optim.Adam, betas=(0.9, 0.99), amsgrad=optimizer_config.amsgrad
        )
    else:
        optimizer_func = partial(torch.optim.Adam, amsgrad=optimizer_config.amsgrad)

    optimizer = OptimWrapper.create(
        optimizer_func,
        3e-3,   # TODO: CHECKING LR HERE !!!
        get_layer_groups(model),
        wd=optimizer_config.wd,
        true_wd=optimizer_config.fixed_wd,
        bn_wd=True,
    )

    return optimizer


def _create_learning_rate_scheduler(optimizer, learning_rate_config, total_step):
    """Create optimizer learning rate scheduler based on config.

    Args:
        learning_rate_config: A LearningRate proto message.

    Returns:
        A learning rate.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    lr_scheduler = None
    learning_rate_type = learning_rate_config.type
    config = learning_rate_config

    if learning_rate_type == "multi_phase":
        lr_phases = []
        mom_phases = []
        for phase_cfg in config.phases:
            lr_phases.append((phase_cfg.start, phase_cfg.lambda_func))
            mom_phases.append((phase_cfg.start, phase_cfg.momentum_lambda_func))
        lr_scheduler = lsf.LRSchedulerStep(optimizer, total_step, lr_phases, mom_phases)
    elif learning_rate_type == "one_cycle":
        lr_scheduler = lsf.OneCycle(
            optimizer,
            total_step,
            config.lr_max,
            config.moms,
            config.div_factor,
            config.pct_start,
        )
    elif learning_rate_type == "exponential_decay":
        lr_scheduler = lsf.ExponentialDecay(
            optimizer,
            total_step,
            config.initial_learning_rate,
            config.decay_length,
            config.decay_factor,
            config.staircase,
        )
    elif learning_rate_type == "manual_stepping":
        lr_scheduler = lsf.ManualStepping(
            optimizer, total_step, config.boundaries, config.rates
        )
    elif lr_scheduler is None:
        raise ValueError("Learning_rate %s not supported." % learning_rate_type)

    return lr_scheduler