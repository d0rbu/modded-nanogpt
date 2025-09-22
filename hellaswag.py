from pathlib import Path

import arguably
import torch as th
import torch.nn.functional as F
import yaml
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from loguru import logger
from transformers import (
    AutoTokenizer,
    CausalLMOutputWithPast,
    PretrainedConfig,
    PreTrainedModel,
)

from train_gpt_medium import GPT, Hyperparameters, get_window_size_blocks_helper, norm


class EvaluationGPT(GPT):
    def forward(
        self, input_ids: th.Tensor, labels: th.Tensor | None = None
    ) -> CausalLMOutputWithPast:
        assert input_ids.ndim == 2
        assert labels is None or labels.ndim == 2

        num_samples = input_ids.shape[0]

        input_ids_flat = input_ids.flatten()
        labels_flat = labels.flatten() if labels is not None else None

        ve = [value_embed(input_ids_flat) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = (
            [ve[0], ve[1], ve[2]]
            + [None] * (len(self.blocks) - 6)
            + [ve[0], ve[1], ve[2]]
        )
        assert len(ve) == len(self.blocks)

        sliding_window_num_blocks = get_window_size_blocks_helper(3584)

        long_bm, short_bm = self.create_blockmasks(
            input_ids_flat, sliding_window_num_blocks
        )
        block_masks = [
            long_bm,
            short_bm,
            short_bm,
            short_bm,
            long_bm,
            short_bm,
            short_bm,
            short_bm,
            short_bm,
            short_bm,
            short_bm,
            long_bm,
            short_bm,
            short_bm,
            short_bm,
            long_bm,
        ]
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(
            self.embed(input_ids_flat)[None]
        )  # use of norm here by @Grad62304977

        skip_connections = []
        skip_map = {
            9: 6,
            10: 4,
            11: 2,
        }
        skip_weights = self.scalars[: len(self.blocks)]
        lambdas = self.scalars[1 * len(self.blocks) : 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks) : 5 * len(self.blocks)].view(
            -1, 2
        )
        for i in range(len(self.blocks)):
            if i in skip_map:
                x = x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]
            x = self.blocks[i](x, ve[i], x0, block_masks[i], lambdas[i], sa_lambdas[i])
            skip_connections.append(x)

        x = norm(x)

        loss = None
        if labels_flat is not None:
            loss = 0
            for i in range(num_samples):
                logits: th.Tensor = F.linear(
                    x.flatten(end_dim=1).chunk(num_samples)[i],
                    self.lm_head_w.bfloat16(),
                ).float()
                loss += (
                    F.cross_entropy(
                        15 * logits * th.rsqrt(logits.square() + 225),
                        labels_flat.chunk(num_samples)[i],
                    )
                    / num_samples
                )

        # just return the probabilities
        output = x.view(*input_ids.shape)
        logits: th.Tensor = F.linear(output, self.lm_head_w.bfloat16()).float()

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class CustomConfig(PretrainedConfig):
    pass


class CustomModel(PreTrainedModel):
    config_class = CustomConfig

    def __init__(self, config: CustomConfig, model_path: Path):
        super().__init__(config)
        hparams = Hyperparameters()

        logger.info(f"Initializing model with hyperparameters: {hparams}")
        self.model = EvaluationGPT(
            vocab_size=hparams.vocab_size,
            num_layers=16,
            num_heads=8,
            model_dim=1024,
            max_seq_len=max(hparams.train_seq_len, hparams.val_seq_len),
        )

        logger.info(f"Loading model state dict from {model_path}")
        self.model.load_state_dict(th.load(model_path))

        logger.info("Initializing tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def forward(self, input_ids: th.Tensor, labels: th.Tensor | None = None):
        return self.model(input_ids, labels)


@arguably.command()
def main(logs_dirpath: str = "logs"):
    logger.info(f"Evaluating logs in {logs_dirpath}")
    logs_dir = Path(logs_dirpath)

    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory {logs_dirpath} does not exist")

    run_dirpaths = set(logs_dir.glob("*/"))

    if len(run_dirpaths) > 1:
        raise ValueError(f"Multiple run directories found in {logs_dirpath}")

    run_dirpath = run_dirpaths.pop()

    logger.info(f"Finding latest model in {run_dirpath}")
    model_path = run_dirpath / "latest_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist")

    logger.info(f"Loading model from {model_path}")
    model = CustomModel(CustomConfig(), model_path)
    logger.info("Wrapping model in HFLM")
    wrapped_model = HFLM(pretrained=model, tokenizer=model.tokenizer)

    logger.info("Evaluating model")
    results = evaluator.evaluate(model=wrapped_model, tasks=["hellaswag"])
    logger.info(f"Saving results to {run_dirpath / 'hellaswag.yaml'}")
    with open(run_dirpath / "hellaswag.yaml", "w") as f:
        yaml.dump(results, f)

    logger.success("hooray :D")


if __name__ == "__main__":
    arguably.run()
