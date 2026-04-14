import argparse
import json
from pathlib import Path


DEFAULT_SIZES = [8, 16, 32, 64, 128, 256]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
        f.write("\n")


def build_experiment_name(base_name: str, size: int) -> str:
    return f"{base_name}_code{size}"


def derive_description(base_description: str, size: int) -> str:
    if "latent=" in base_description:
        prefix = base_description.split("latent=")[0].rstrip()
        return f"{prefix} latent={size}"
    return f"{base_description} latent={size}"


def prepare_experiment_specs(base_experiment_dir: Path, size: int, dry_run: bool):
    base_specs_path = base_experiment_dir / "specs.json"
    specs = load_json(base_specs_path)

    experiment_name = build_experiment_name(base_experiment_dir.name, size)
    target_dir = base_experiment_dir.parent / experiment_name
    target_specs_path = target_dir / "specs.json"

    specs["CodeLength"] = size
    specs["Description"] = derive_description(specs.get("Description", base_experiment_dir.name), size)

    if dry_run:
        print(f"[dry-run] would write {target_specs_path}")
    else:
        dump_json(target_specs_path, specs)

    return experiment_name, target_dir


def prepare_encoder_config(base_config_path: Path, config_stem: str, size: int, dry_run: bool):
    config = load_json(base_config_path)

    target_path = base_config_path.parent / f"{config_stem}.json"
    config["checkpoint_file"] = f"_{config_stem}_best_model.pt"

    if dry_run:
        print(f"[dry-run] would write {target_path}")
    else:
        dump_json(target_path, config)

    return target_path


def main():
    parser = argparse.ArgumentParser(
        description="Create a batch of DeepSDF CodeLength ablation experiment folders and matching encoder configs."
    )
    parser.add_argument(
        "--base-experiment-dir",
        required=True,
        help="Path to the baseline DeepSDF experiment directory that contains specs.json.",
    )
    parser.add_argument(
        "--base-config",
        default=None,
        help="Optional encoder config json to clone alongside each CodeLength setting.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=DEFAULT_SIZES,
        help="Latent dimensions to generate. Default: 8 16 32 64 128 256",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be generated without writing them.",
    )
    args = parser.parse_args()

    base_experiment_dir = Path(args.base_experiment_dir).resolve()
    if not (base_experiment_dir / "specs.json").exists():
        raise FileNotFoundError(f"Cannot find specs.json under {base_experiment_dir}")

    base_config_path = Path(args.base_config).resolve() if args.base_config else None
    if base_config_path and not base_config_path.exists():
        raise FileNotFoundError(f"Cannot find config file: {base_config_path}")

    print("Preparing CodeLength ablation:")
    print(f"  base experiment: {base_experiment_dir}")
    if base_config_path:
        print(f"  base config:     {base_config_path}")
    print(f"  sizes:           {args.sizes}")

    for size in args.sizes:
        experiment_name, target_dir = prepare_experiment_specs(base_experiment_dir, size, args.dry_run)
        print(f"- experiment {size}: {target_dir}")

        if base_config_path:
            config_stem = experiment_name
            config_path = prepare_encoder_config(base_config_path, config_stem, size, args.dry_run)
            print(f"  config      {size}: {config_path}")


if __name__ == "__main__":
    main()
