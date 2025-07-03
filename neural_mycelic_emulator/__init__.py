"""Neural-Mycelic Emulator package.

High-level layout
-----------------
preprocessor/   spike-to-glyph pipeline
models/         emulator architectures + training helpers
docs/           design letters & research notes

Import shortcuts
----------------
>>> from neural_mycelic_emulator.preprocessor.pipeline import tsv_to_glyph_sequences
"""

from importlib import metadata

__all__ = [
    "__version__",
]

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:  # local checkout
    __version__ = "0.0.0.dev0" 