from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("embeddings_extract")
def set_cfg_embeddings_extract(cfg):
    """Reconfigure the default config value for embeddings extraction.

    Returns:
        Reconfigured embeddings extraction configuration use by the experiment.
    """
    cfg.embeddings = CN()
    cfg.embeddings.type = "logits"
