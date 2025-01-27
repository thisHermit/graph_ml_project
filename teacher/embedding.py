#!/usr/bin/env python

import datetime
import os
import torch
import logging
import graphgps  # noqa, register custom modules

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg,
    dump_cfg,
    set_cfg,
    load_cfg,
)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric import seed_everything
from torch_geometric import nn as nng

from graphgps.finetuning import load_pretrained_model_cfg, init_model_from_pretrained
from graphgps.logger import create_logger

from typing import Dict, Iterable, Callable


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


class EmbeddingsExtractor(torch.nn.Module):
    """
    Extract embeddings from a model at specified layers.
    """

    def __init__(self, model: torch.nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output

        return fn

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _ = self.model(x)
        return self._features


def get_embeddings(model, loaders, cfg):
    with torch.no_grad():
        model.eval()
        logging.info("Total number of nodes: " + str(cfg.dataset.num_nodes))
        logging.info("Total number of graphs: " + str(cfg.dataset.num_graphs))
        # Check if embeddings type is node
        if cfg.embeddings.type and cfg.embeddings.type == "node":
            # Initialize embeddings tensor and nodes count on CPU first
            node_embeddings = torch.zeros(
                cfg.dataset.num_nodes, int(cfg.gt.dim_hidden), dtype=torch.float32
            )
            summary_embeddings = torch.zeros(
                cfg.dataset.num_graphs, int(cfg.gt.dim_hidden), dtype=torch.float32
            )
            nodes_count = torch.zeros(cfg.dataset.num_graphs, 1)

            # Setup extractor for final layer embeddings
            node_embedding_extractor = EmbeddingsExtractor(
                model, ["model.layers.4.ff_dropout2"]
            )

            device = torch.device(cfg.accelerator)
            # Loop over loaders (train, val, test)
            for loader_idx, loader in enumerate(loaders):
                logging.info(f"Processing loader {loader_idx + 1}/{len(loaders)}")

                for batch_idx, batch in enumerate(loader):
                    batch = batch.to(device)
                    logging.info(
                        f"Processing Batch {batch_idx + 1}/{len(loader)}, "
                        f"size: {batch.y.shape[0]}"
                    )

                    with torch.amp.autocast("cuda", enabled=True):  # Mixed precision
                        # Get final layerembeddings for current batch
                        h_embeddings = list(node_embedding_extractor(batch).values())[0]

                        # Process batch indices
                        ptr = batch.ptr
                        new_pre, new_post = ptr[:-1], ptr[1:]
                        # Get graph ids for current batch
                        batch_wrt_graph = torch.tensor(
                            [batch.graph_id[g_id].item() for g_id in batch.batch]
                        )
                        # Update node embeddings
                        row_ids = batch.node_idx
                        node_embeddings[row_ids] = h_embeddings.float().cpu()
                        # Update summary embeddings
                        unique_graphs = batch_wrt_graph.unique_consecutive()
                        summary = nng.global_mean_pool(h_embeddings, batch.batch)
                        summary_embeddings[unique_graphs] = (
                            summary.float().cpu()
                        )  # Convert to float32

                        # Update nodes count
                        len_idx = torch.tensor(
                            [
                                (post - pre).item()
                                for pre, post in list(zip(new_pre, new_post))
                            ]
                        )
                        nodes_count[unique_graphs] = len_idx.float().unsqueeze(-1)

                    # Clear cache after each batch
                    torch.cuda.empty_cache()

            # Calculate ptr on CPU - cumulative sum of node count
            ptr = torch.cat([torch.tensor([0]), torch.cumsum(nodes_count, dim=0)[:, 0]])

            # Save results
            data_dict = {
                "h-embeddings": node_embeddings,
                "g-embeddings": summary_embeddings,
                "ptr": ptr,
            }

        else:
            # Initialize logits embeddings on CPU with explicit dtype
            logits_embeddings = torch.zeros(
                cfg.dataset.num_graphs, int(cfg.dataset.num_tasks), dtype=torch.float32
            )

            device = torch.device(cfg.accelerator)
            for loader_idx, loader in enumerate(loaders):
                logging.info(f"Processing loader {loader_idx + 1}/{len(loaders)}")

                for batch_idx, batch in enumerate(loader):
                    logging.info(
                        f"Processing Batch {batch_idx + 1}/{len(loader)}, "
                        f"size: {batch.y.shape[0]}"
                    )
                    batch = batch.to(device)

                    with torch.amp.autocast("cuda", enabled=True):
                        # Get logits embeddings for current batch
                        z_embeddings, _ = model(batch)
                        # Get graph ids for current batch
                        batch_wrt_graph = torch.tensor(
                            [batch.graph_id[g_id].item() for g_id in batch.batch]
                        )
                        # Get unique graph ids for current batch
                        unique_graphs = batch_wrt_graph.unique_consecutive()
                        # Update logits embeddings for current batch
                        logits_embeddings[unique_graphs] = z_embeddings.float().cpu()

                    torch.cuda.empty_cache()

            data_dict = {"logits": logits_embeddings}

        # Save to disk
        save_path = os.path.join(
            cfg.run_dir, f"{cfg.embeddings.type.lower()}-embeddings.pt"
        )
        torch.save(data_dict, save_path)
        logging.info(f"Embeddings saved to {save_path}")


if __name__ == "__main__":
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    run_id = 0
    set_printing()
    seed_everything(cfg.seed)
    auto_select_device()
    if cfg.pretrained.dir:
        cfg = load_pretrained_model_cfg(cfg)
    logging.info(
        f"[*] Run ID {run_id}: seed={cfg.seed}, "
        f"split_index={cfg.dataset.split_index}"
    )
    if cfg.embeddings.type and cfg.embeddings.type == "node":
        print("Extracting node embeddings")
    elif cfg.embeddings.type and cfg.embeddings.type == "logits":
        print("Extracting logits embeddings")
    logging.info(f"    Starting now: {datetime.datetime.now()}")
    loaders = create_loader()
    model = create_model()
    loggers = create_logger()
    if cfg.pretrained.dir:
        # Load pretrained model provided by the user in config file
        model = init_model_from_pretrained(
            model,
            cfg.pretrained.dir,
            cfg.pretrained.freeze_main,
            cfg.pretrained.reset_prediction_head,
            seed=cfg.seed,
        )
    # Extract embeddings
    get_embeddings(model, loaders, cfg)
