import argparse

# Base protein language model. The revision is pinned because the model code is
# loaded from the Hugging Face Hub (trust_remote_code) and later revisions changed
# the sequence-classification pooling and the required library versions. The LoRA
# adapter in ATHENA_IDP_model_params was trained against this revision.
DEFAULT_BASE_MODEL = "Synthyra/ESMplusplus_small"
DEFAULT_BASE_MODEL_REVISION = "1bb4632b2b9ed1ee11718b3dee62871a097e5663"

# Pooling the classification head was trained with: [CLS] embedding + mean embedding.
EXPECTED_POOLING_TYPES = ("cls", "mean")

OUTPUT_TYPE_PROBABILITY = "IDP_probability"
OUTPUT_TYPE_LOGITS = "before_softmax"
OUTPUT_TYPES = (OUTPUT_TYPE_PROBABILITY, OUTPUT_TYPE_LOGITS)


class Config:
    def __init__(self, args):
        # Basic settings
        self.output_type = args.output_type
        self.base_model = args.base_model
        self.base_model_revision = getattr(
            args, "base_model_revision", DEFAULT_BASE_MODEL_REVISION
        )
        self.classifier_params_path = args.classifier_params_path
        self.fasta_path = args.fasta_path
        self.output_dir = args.output_dir

        # Adapter settings
        self.adapter_paths = args.adapter_paths
        self.adapter_weights = args.adapter_weights

        # Model settings
        self.num_labels = args.num_labels
        self.max_length = args.max_length
        self.batch_size = args.batch_size

        # Run settings
        self.title = args.title


class ParseDictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        d = {}
        for item in values:
            try:
                key, value = item.split('=', 1)
                d[key] = value
            except ValueError:
                raise argparse.ArgumentError(
                    self, f"Argument must be in 'key=value' format: '{item}'"
                )
        setattr(namespace, self.dest, d)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Protein-level IDP classification with a LoRA fine-tuned protein language model"
    )

    # --- Basic settings ---
    parser.add_argument(
        "--output_type", type=str, default=OUTPUT_TYPE_PROBABILITY,
        choices=OUTPUT_TYPES,
        help=(
            f"Output type: '{OUTPUT_TYPE_PROBABILITY}' (softmax probability of the IDP class) "
            f"or '{OUTPUT_TYPE_LOGITS}' (raw logits, the ATHENA score). "
            f"Default: {OUTPUT_TYPE_PROBABILITY}"
        )
    )
    parser.add_argument(
        "--base_model", type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Hugging Face name of the base model. Default: {DEFAULT_BASE_MODEL}"
    )
    parser.add_argument(
        "--base_model_revision", type=str,
        default=DEFAULT_BASE_MODEL_REVISION,
        help=(
            "Hugging Face revision (commit hash) of the base model. Pinned to the revision "
            f"the adapter was trained with. Default: {DEFAULT_BASE_MODEL_REVISION}"
        )
    )
    parser.add_argument(
        "--classifier_params_path", type=str,
        default=None,
        help=(
            "Deprecated and ignored. The trained classifier head is stored inside the "
            "LoRA adapter (adapter_model.safetensors, PEFT modules_to_save); "
            "classifier_params.pth is not used."
        )
    )
    parser.add_argument(
        "--fasta_path", type=str,
        default="input/example_sequences.fasta",
        help="Path to the input FASTA file"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="output",
        help="Directory where prediction results are saved"
    )

    # --- Adapter settings ---
    parser.add_argument(
        "--adapter_paths",
        nargs='+',
        action=ParseDictAction,
        metavar="NAME=PATH",
        required=True,
        help=(
            "Adapter name and path in 'name=/path/to/adapter' format (multiple allowed). "
            "Required: the adapter also contains the trained classifier head."
        )
    )
    parser.add_argument(
        "--adapter_weights",
        nargs='+',
        type=float,
        help="Weight of each adapter, space-separated (e.g. 0.7 0.3)"
    )

    # --- Model settings ---
    parser.add_argument(
        "--num_labels", type=int, default=2,
        help="Number of classification labels. Default: 2"
    )
    parser.add_argument(
        "--max_length", type=int, default=10000,
        help="Maximum sequence length for the tokenizer. Default: 10000"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for prediction. Default: 32"
    )

    # --- Output settings ---
    parser.add_argument(
        "--title", type=str, default="test",
        help="Title of the run (used in output file names)"
    )

    args = parser.parse_args()

    # Consistency check between adapter_paths and adapter_weights
    if args.adapter_paths and args.adapter_weights:
        if len(args.adapter_paths) != len(args.adapter_weights):
            parser.error("The number of --adapter_paths and --adapter_weights must match.")

    return args
