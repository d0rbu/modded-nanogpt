from pathlib import Path

import arguably
from lm_eval.api.model import LM
from lm_eval.tasks import get_task_dict


@arguably.command()
def main(logs_dirpath: str = "logs"):
    logs_dir = Path(logs_dirpath)

    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory {logs_dirpath} does not exist")

    run_dirpaths = set(logs_dir.glob("*/"))

    if len(run_dirpaths) > 1:
        raise ValueError(f"Multiple run directories found in {logs_dirpath}")

    run_dirpath = run_dirpaths.pop()

    model_path = run_dirpath / "latest_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist")

    raise NotImplementedError("Load from model_path into an LM-compatible format")
    LM
    get_task_dict


if __name__ == "__main__":
    arguably.run()
