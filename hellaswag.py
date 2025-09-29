from pathlib import Path

print("importing torch...")

import arguably
import torch as th
import torch.nn.functional as F
import yaml

print("importing lm_eval...")

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from loguru import logger

logger.info("importing transformers...")

from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

logger.info("importing train_gpt_medium...")

from train_gpt_medium import (
    GPT,
    args,
    get_window_size_blocks_helper,
    next_multiple_of_n,
    norm,
)


class EvaluationGPT(GPT):
    def forward(
        self, input_ids: th.Tensor, labels: th.Tensor | None = None
    ) -> CausalLMOutputWithPast:
        assert input_ids.ndim == 2
        assert labels is None or labels.ndim == 2
        if input_ids.dtype not in (th.int32, th.int64):
            logger.warning(f"Input IDs dtype is {input_ids.dtype}, converting to int32")
            input_ids = input_ids.to(dtype=th.int32)

        if input_ids.device != self.embed.weight.device:
            logger.warning(
                f"Input IDs device is {input_ids.device}, converting to {self.embed.weight.device}"
            )
            input_ids = input_ids.to(device=self.embed.weight.device)

        input_ids_with_eos_separators = th.cat(
            [
                input_ids,
                th.full(
                    (input_ids.shape[0], 1),
                    50256,
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                ),
            ],
            dim=1,
        )

        del input_ids

        num_samples = input_ids_with_eos_separators.shape[0]

        input_ids_flat_unpadded = input_ids_with_eos_separators.flatten()
        unpadded_len = input_ids_flat_unpadded.shape[0]
        padded_len = next_multiple_of_n(unpadded_len, n=128)
        padding_len = padded_len - unpadded_len

        input_ids_flat = th.cat(
            [
                input_ids_flat_unpadded,
                th.full(
                    (padding_len,),
                    50256,
                    device=input_ids_flat_unpadded.device,
                    dtype=input_ids_flat_unpadded.dtype,
                ),
            ],
            dim=0,
        )

        del input_ids_flat_unpadded

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

        unpadded_outputs = x[:, :unpadded_len]
        unrolled_outputs = unpadded_outputs.view(*input_ids_with_eos_separators, -1)
        outputs = unrolled_outputs[:, :-1]

        loss = None
        if labels_flat is not None:
            loss = 0
            for i in range(num_samples):
                logits: th.Tensor = F.linear(
                    outputs.flatten(end_dim=1).chunk(num_samples)[i],
                    self.lm_head_w.bfloat16(),
                ).float()
                loss += (
                    F.cross_entropy(
                        15 * logits * th.rsqrt(logits.square() + 225),
                        labels_flat.chunk(num_samples)[i],
                    )
                    / num_samples
                )

        logits: th.Tensor = F.linear(outputs, self.lm_head_w.bfloat16()).float()

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

        logger.info(f"Initializing model with hyperparameters: {args}")
        self.model = EvaluationGPT(
            vocab_size=args.vocab_size,
            num_layers=16,
            num_heads=8,
            model_dim=1024,
            max_seq_len=max(args.train_seq_len, args.val_seq_len),
        )

        logger.info(f"Loading model state dict from {model_path}")
        model_state_dict = th.load(model_path)

        logger.info("Renaming keys in model state dict")
        renamed_model_state_dict = {
            key.replace("_orig_mod.", ""): value
            for key, value in model_state_dict.items()
            if key.startswith("_orig_mod.")
        }

        self.model.load_state_dict(renamed_model_state_dict)

        del model_state_dict, renamed_model_state_dict

        logger.info("Initializing tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def forward(self, input_ids: th.Tensor, labels: th.Tensor | None = None):
        return self.model(input_ids, labels)


@arguably.command()
def main(logs_dirpath: str = "logs"):
    # Single GPU setup
    assert th.cuda.is_available()
    device = th.device("cuda", 0)
    th.cuda.set_device(device)

    logger.info(f"Evaluating logs in {logs_dirpath}")
    logs_dir = Path(logs_dirpath)

    assert logs_dir.exists(), f"Logs directory {logs_dirpath} does not exist"

    run_dirpaths = set(child for child in logs_dir.iterdir() if child.is_dir())

    assert len(run_dirpaths) <= 1, f"Multiple run directories found in {logs_dirpath}"
    assert len(run_dirpaths) > 0, f"No run directories found in {logs_dirpath}"

    run_dirpath = run_dirpaths.pop()

    logger.info(f"Finding latest model in {run_dirpath}")
    model_path = run_dirpath / "latest_model.pt"
    assert model_path.exists(), f"Model path {model_path} does not exist"

    logger.info(f"Loading model from {model_path}")
    model = CustomModel(CustomConfig(), model_path)
    logger.info("Wrapping model in HFLM")
    wrapped_model = HFLM(pretrained=model, tokenizer=model.tokenizer)

    logger.info("Evaluating model")
    results = evaluator.simple_evaluate(model=wrapped_model, tasks=["hellaswag"])
    logger.info(f"Saving results to {run_dirpath / 'hellaswag.yaml'}")
    with open(run_dirpath / "hellaswag.yaml", "w") as f:
        yaml.dump(results, f)

    logger.success("hooray :D")


if __name__ == "__main__":
    arguably.run()
