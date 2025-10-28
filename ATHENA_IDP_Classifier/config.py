import argparse

class Config:
    def __init__(self, args):
        # 基本設定
        self.output_type = args.output_type
        self.base_model = args.base_model
        self.classifier_params_path = args.classifier_params_path
        self.fasta_path = args.fasta_path
        self.output_dir = args.output_dir
        
        # アダプター設定
        self.adapter_paths = args.adapter_paths
        self.adapter_weights = args.adapter_weights
        
        # モデル設定
        self.num_labels = args.num_labels
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        
        # タスク設定      
        self.title = args.title

class ParseDictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        d = {}
        for item in values:
            try:
                key, value = item.split('=', 1)
                d[key] = value
            except ValueError:
                raise argparse.ArgumentError(self, f"引数は 'key=value' 形式である必要があります: '{item}'")
        setattr(namespace, self.dest, d)

def parse_args():
    parser = argparse.ArgumentParser(
        description="xLoRAを用いたタンパク質配列の埋め込み生成・予測"
    )

    # --- 基本設定 ---
    parser.add_argument(
        "--output_type", type=str, default="IDP_probability",
        help="出力タイプを指定"
    )
    parser.add_argument(
        "--base_model", type=str,
        default="Synthyra/ESMplusplus_small",
        help="HuggingFaceのベースモデル名"
    )
    parser.add_argument(
        "--classifier_params_path", type=str,
        default="ATHENA_IDP_model_params",
        help="訓練済み分類器の重みファイルを含むディレクトリ"
    )
    parser.add_argument(
        "--fasta_path", type=str, 
        default="input/example_sequences.fasta",
        help="入力FASTAファイルへのパス"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="output",
        help="予測結果を保存するディレクトリ"
    )

    # --- アダプター設定 ---
    parser.add_argument(
        "--adapter_paths",
        nargs='+',
        action=ParseDictAction,
        metavar="NAME=PATH",
        help="アダプター名とパスを 'name=/path/to/adapter' 形式で指定 (複数可)"
    )
    parser.add_argument(
        "--adapter_weights",
        nargs='+',
        type=float,
        help="各アダプターの重みをスペース区切りで指定 (例: 0.7 0.3)"
    )

    # --- モデル設定 ---
    parser.add_argument(
        "--num_labels", type=int, default=2,
        help="分類のラベル数。デフォルト: 2"
    )
    parser.add_argument(
        "--max_length", type=int, default=10000,
        help="トークナイザーの最大配列長。デフォルト: 2048"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="予測のバッチサイズ。デフォルト: 32"
    )

    # --- 出力設定 ---
    parser.add_argument(
        "--title", type=str, default="test",
        help="実行のタイトル (保存ファイル名などに使用)"
    )

    args = parser.parse_args()

    # adapter_paths と adapter_weights の整合性チェック
    if args.adapter_paths and args.adapter_weights:
        if len(args.adapter_paths) != len(args.adapter_weights):
            parser.error("--adapter_paths の数と --adapter_weights の数が一致しません。")

    return args