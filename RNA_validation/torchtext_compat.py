"""Small pure-Python TorchText vocabulary compatibility layer for scGPT.

TorchText ended at PyTorch 2.3 and cannot be loaded with the newer ARM64 PyTorch used
on TACC. scGPT's embedding path only needs the vocabulary container and factory below.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import sys
import types


class Vocab:
    def __init__(self, source=None):
        if isinstance(source, Vocab):
            tokens = source.get_itos()
        elif isinstance(source, Mapping):
            tokens = list(source)
        elif source is None:
            tokens = []
        elif hasattr(source, "get_itos"):
            tokens = list(source.get_itos())
        else:
            raise TypeError(f"Unsupported vocabulary source: {type(source)!r}")
        self._itos = list(tokens)
        self._reindex()
        self._default_index = None
        # scGPT passes the internal `.vocab` object back to the Vocab constructor.
        self.vocab = self

    def _reindex(self) -> None:
        self._stoi = {token: index for index, token in enumerate(self._itos)}

    def __len__(self) -> int:
        return len(self._itos)

    def __contains__(self, token: object) -> bool:
        return token in self._stoi

    def __getitem__(self, token: str) -> int:
        if token in self._stoi:
            return self._stoi[token]
        if self._default_index is not None:
            return self._default_index
        raise RuntimeError(f"Token {token!r} not found and default index is not set")

    def __call__(self, tokens: Iterable[str]) -> list[int]:
        return self.lookup_indices(tokens)

    def append_token(self, token: str) -> None:
        if token in self:
            raise RuntimeError(f"Token {token!r} already exists")
        self._itos.append(token)
        self._stoi[token] = len(self._itos) - 1

    def insert_token(self, token: str, index: int) -> None:
        if token in self:
            raise RuntimeError(f"Token {token!r} already exists")
        if index < 0 or index > len(self._itos):
            raise RuntimeError(f"Invalid insertion index: {index}")
        self._itos.insert(index, token)
        self._reindex()

    def set_default_index(self, index: int | None) -> None:
        self._default_index = index

    def get_default_index(self) -> int | None:
        return self._default_index

    def get_stoi(self) -> dict[str, int]:
        return dict(self._stoi)

    def get_itos(self) -> list[str]:
        return list(self._itos)

    def lookup_indices(self, tokens: Iterable[str]) -> list[int]:
        return [self[token] for token in tokens]

    def lookup_token(self, index: int) -> str:
        return self._itos[index]

    def lookup_tokens(self, indices: Iterable[int]) -> list[str]:
        return [self.lookup_token(index) for index in indices]


def vocab(ordered_dict: Mapping[str, int], min_freq: int = 1, specials=None, special_first: bool = True) -> Vocab:
    tokens = [token for token, frequency in ordered_dict.items() if frequency >= min_freq]
    specials = list(specials or [])
    tokens = (specials + tokens) if special_first else (tokens + specials)
    return Vocab(dict.fromkeys(tokens))


def install_torchtext_compat() -> bool:
    """Install the shim only when the real TorchText vocabulary cannot be imported."""
    try:
        import torchtext.vocab  # noqa: F401
        return False
    except (ImportError, OSError):
        for name in [key for key in sys.modules if key == "torchtext" or key.startswith("torchtext.")]:
            sys.modules.pop(name, None)
        package = types.ModuleType("torchtext")
        module = types.ModuleType("torchtext.vocab")
        module.Vocab = Vocab
        module.vocab = vocab
        package.vocab = module
        sys.modules["torchtext"] = package
        sys.modules["torchtext.vocab"] = module
        return True
