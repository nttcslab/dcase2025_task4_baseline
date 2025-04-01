import os
import sys
import datetime
import logging
from typing import Dict, List, NoReturn
import yaml
import importlib
import pytz
import torch

LABELS = {
    'dcase2025t4': ["AlarmClock", "BicycleBell", "Blender", "Buzzer",
                    "Clapping", "Cough", "CupboardOpenClose", "Dishes",
                    "Doorbell", "FootSteps", "HairDryer", "MechanicalFans",
                    "MusicalKeyboard", "Percussion", "Pour", "Speech",
                    "Typing", "VacuumCleaner"],
}





def ignore_warnings():
    import warnings

    # Filter out warnings containing "deprecated"
    warnings.filterwarnings("ignore", message=".*invalid value encountered in cast.*")
    warnings.filterwarnings("ignore", message=".*deprecated.*")
    warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
    warnings.filterwarnings("ignore", message=".*that has Tensor Cores.*")
    warnings.filterwarnings("ignore", message=".*cudnnFinalize Descriptor Failed.*")
    warnings.filterwarnings("ignore", message=".*torch.use_deterministic_algorithms(True, warn_only=True)*")
    from transformers import logging
    logging.set_verbosity_error()

logging_levels = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET
}
def logging_setup(directory, file_log_level = "INFO", console_log_level = 'DEBUG', timezone='Asia/Tokyo'):
    file_log_level = logging_levels[file_log_level]
    console_log_level = logging_levels[console_log_level]
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # create directory, generate logfilename
    os.makedirs(directory, exist_ok=True)
    a = datetime.datetime.now().astimezone(pytz.timezone(timezone))
    logfilename='%04d%02d%02d_%02dh%02d.log'%(a.year, a.month, a.day, a.hour, a.minute)
    log_full_path = os.path.join(directory, logfilename)

    # setup log to file
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    file_handler = logging.FileHandler(log_full_path)
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # print to console
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s (%(processName)s): %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level) # always debug level
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)


def parse_yaml(config_yaml: str) -> Dict:
    r"""Parse yaml file.

    Args:
        config_yaml (str): config yaml path

    Returns:
        yaml_dict (Dict): parsed yaml file
    """

    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)

def initialize_config(module_cfg, reload=False):
    if reload and module_cfg["module"] in sys.modules:
        module = importlib.reload(sys.modules[module_cfg["module"]])
    else: module = importlib.import_module(module_cfg["module"])
    if 'args' in module_cfg.keys(): return getattr(module, module_cfg["main"])(**module_cfg["args"])
    return getattr(module, module_cfg["main"])()

def lightning_load_from_checkpoint(lightning_module_cfg, ckpt_path):
    module = importlib.import_module(lightning_module_cfg["module"])
    model_class = getattr(module, lightning_module_cfg["main"])
    model = model_class.load_from_checkpoint(ckpt_path, **lightning_module_cfg['args'])
    return model


def ca_metric(est_lb, est_wf, ref_lb, ref_wf, mixture, metricfunc):
    """
    Permutation-sensitive metric calculation.

    Args:
        est_lb (list of str): Labels for estimated sources, length n_est_sources.
        est_wf (torch.Tensor): Corresponding estimated waveforms, shape [n_est_sources, wf_len].
        ref_lb (list of str): Labels for reference sources, length n_ref_sources.
        ref_wf (torch.Tensor): Corresponding reference waveforms, shape [n_ref_sources, wf_len].
        mixture (torch.Tensor): Mixture waveform at the reference channel, shape [wf_len].
        metricfunc (callable): metric function, metricfunc(predicted, target)
    """

    est_dict = {lb:wf for lb, wf in zip(est_lb, est_wf)}
    ref_dict = {lb:wf for lb, wf in zip(ref_lb, ref_wf)}

    tp_lb = set(est_lb) & set(ref_lb) - {'silence'} # true positive labels
    fp_lb = set(est_lb) - tp_lb - {'silence'} # false positive labels
    fn_lb = set(ref_lb) - tp_lb - {'silence'} # false negative labels

    result = torch.tensor([])
    if tp_lb:
        tp_est_wf = torch.stack([est_dict[lb] for lb in tp_lb]) # shape [len(tp_lb), wf_len]
        tp_ref_wf = torch.stack([ref_dict[lb] for lb in tp_lb]) # shape [len(tp_lb), wf_len]
        repeated_mixture = mixture.unsqueeze(0).repeat(len(tp_lb), 1)
        score_est = metricfunc(tp_est_wf, tp_ref_wf)
        score_mixture = metricfunc(repeated_mixture, tp_ref_wf)
        improvement = score_est - score_mixture # shape [len(tp_lb)]
        result = torch.cat((result, improvement), dim=0)
    pens =torch.zeros(len(fp_lb) + len(fn_lb))
    result = torch.cat((result, pens), dim=0)

    return result.mean().item()
