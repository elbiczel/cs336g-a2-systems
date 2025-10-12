from dataclasses import dataclass


@dataclass
class TransformerConfig:
    name: str
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float


def small() -> TransformerConfig:
    return TransformerConfig(
        name="S",
        vocab_size=10_000,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        rope_theta=10000.0,
    )


def med() -> TransformerConfig:
    return TransformerConfig(
        name="M",
        vocab_size=10_000,
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=4098,
        rope_theta=10000.0,
    )


def large() -> TransformerConfig:
    return TransformerConfig(
        name="L",
        vocab_size=10_000,
        d_model=1280,
        num_layers=36,
        num_heads=20,
        d_ff=5120,
        rope_theta=10000.0,
    )


def xl() -> TransformerConfig:
    return TransformerConfig(
        name="XL",
        vocab_size=10_000,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
        rope_theta=10000.0,
    )


def two_seven_b() -> TransformerConfig:
    return TransformerConfig(
        name="2p7",
        vocab_size=10_000,
        d_model=2560,
        num_layers=32,
        num_heads=32,
        d_ff=10240,
        rope_theta=10000.0,
    )
