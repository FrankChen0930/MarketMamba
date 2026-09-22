"""Side-effect-free configuration for the isolated V7 integrated candidate."""

from dataclasses import asdict, dataclass
from pathlib import Path
import ast
import hashlib
import json


# Frozen literal copied from V6/marketmamba/config.py.  Keeping it here avoids
# that module's import-time directory creation while making schema drift fail.
V6_FEATURE_GROUPS = {
    "price_momentum": ("Open", "High", "Low", "Close", "Volume", "Return_1d", "Return_5d", "Return_20d", "MA_20", "MA_60", "RSI_14", "ATR_14", "RS_5d", "RS_20d", "RS_60d"),
    "institutional_flow": ("Foreign_Buy", "Foreign_Sell", "Foreign_Net", "Investment_Trust_Net", "Dealer_Net", "Margin_Purchase", "Margin_Repay", "Short_Sale", "Short_Cover", "Margin_Balance", "Short_Balance", "Day_Trade_Volume", "KD_K", "KD_D", "OBV", "Volatility_20d", "Holdings_Large_Pct", "Holdings_Large_Change", "Securities_Balance", "Foreign_Holding_Pct"),
    "fundamentals": ("PER", "PBR", "Revenue_MoM", "Revenue_YoY", "EPS", "EPS_Surprise", "Gross_Margin", "ROE", "Market_Cap_Log", "Book_Value", "Dividend_Yield_Fwd", "Free_Cash_Flow"),
    "macro_environment": ("TWII_Return", "SPX_Return", "VIX", "TNX", "Gold_Return", "Oil_Return", "USD_TWD", "Futures_OI_Foreign", "Options_PC_Ratio", "Fear_Greed", "Business_Signal", "FED_Rate"),
}
V6_FEATURE_COLUMNS = tuple(column for group in V6_FEATURE_GROUPS.values() for column in group)


def validate_v6_feature_literal(source_path: str | Path) -> None:
    """Compare against config.py's literal without importing its mkdir side effects."""
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    node = next((item.value for item in tree.body if isinstance(item, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "FEATURE_GROUPS"
                         for target in item.targets)), None)
    if node is None:
        raise ValueError("FEATURE_GROUPS literal is absent")
    try:
        observed = ast.literal_eval(node)
    except (ValueError, TypeError) as exc:
        raise ValueError("FEATURE_GROUPS must remain an AST literal") from exc
    normalized = {name: tuple(columns) for name, columns in observed.items()}
    if normalized != V6_FEATURE_GROUPS:
        raise ValueError("frozen V6 FEATURE_GROUPS literal has drifted")


@dataclass(frozen=True)
class V7IntegratedConfig:
    input_dim: int = 59
    group_dims: tuple[int, int, int, int] = (15, 20, 12, 12)
    d_model: int = 32
    d_state: int = 8
    expand: int = 2
    head_dim: int = 8
    n_groups: int = 1
    sequence_length: int = 60
    temporal_layers: int = 1
    dropout: float = 0.0
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_floor: float = 1e-4
    a_min: float = 1.0
    a_max: float = 16.0
    chunk_size: int = 64
    fusion_limit: float = 0.90
    horizons: tuple[int, int] = (5, 10)
    predictive_baseline: str = "v2_kg_nomacro"
    economic_baseline: str = "v2_kg_nomacro_f20"
    train_start: str = "2013-01-01"
    train_cutoff: str = "2023-12-31"
    validation_end: str = "2026-09-11"
    label_horizon_days: int = 10
    embargo_trading_days: int = 20

    def validate(self) -> None:
        if len(self.group_dims) != 4 or any(size <= 0 for size in self.group_dims):
            raise ValueError("four positive factor group dimensions are required")
        if sum(self.group_dims) != self.input_dim or self.input_dim != 59:
            raise ValueError("group_dims must sum to input_dim")
        if min(self.d_model, self.d_state, self.expand, self.head_dim,
               self.sequence_length, self.temporal_layers, self.chunk_size) <= 0:
            raise ValueError("model dimensions and lengths must be positive")
        inner = self.expand * self.d_model
        if inner % self.head_dim:
            raise ValueError("expand*d_model must be divisible by head_dim")
        if self.n_groups != 1:
            raise ValueError("grouped RMS/SSD is not supported; n_groups must equal 1")
        if not (0.0 < self.dt_min < self.dt_max):
            raise ValueError("dt bounds must be positive and ordered")
        if not (2.0 / 3.0 <= self.fusion_limit < 1.0):
            raise ValueError("fusion_limit must be in [2/3, 1)")
        if self.horizons != (5, 10):
            raise ValueError("frozen heads must be exactly 5d and 10d")

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DataProtocol:
    """Frozen V6 date, feature and label contract."""

    stock_id_column: str = "stock_id"
    date_column: str = "Date"
    label_columns: tuple[str, str] = ("Alpha_5d", "Alpha_10d")
    required_splits: tuple[str, str] = ("train", "validation")
    purge_days: int = 20
    total_purge_rows: int = 30
    train_start: str = "2013-01-01"
    train_cutoff: str = "2023-12-31"
    validation_end: str = "2026-09-11"
    feature_count: int = 59


def default_manifest_path() -> Path:
    return Path(__file__).with_name("v7_integrated_environment.json")


def get_test_config() -> V7IntegratedConfig:
    cfg = V7IntegratedConfig()
    cfg.validate()
    return cfg
