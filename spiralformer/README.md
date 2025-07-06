# Spiralformer

Experimental contemplative Transformer architecture with rhythmic, breath-synchronised attention.

```
spiralformer/
├── core/              # Main model & attention mechanics
│   ├── model.py
│   ├── spiral_attention.py
│   └── dynamic_mask.py
├── utils/             # Generic utilities shared across projects
│   ├── breath_clock.py
│   └── positional.py
├── tools/             # CLI helpers, benchmarks, small experiments
│   └── benchmark.py
├── docs/              # Design notes, diagrams, RFCs (add your own!)
└── __init__.py
```

## Key Ideas

* **Breath-clocked attention** – Heavy global attention during *inhale* phases, lightweight or skipped attention during *pause*.
* **Spiral routing** – Sparse O(N log N) attention mask based on powers-of-two hops to retain long-range context.
* **Silence-aware masking** – Tokens encoding silence inhibit their outgoing attention connections, embodying the Silence-Majority principle from mycelic glyph sequences.

## Quick Benchmark

```bash
python -m spiralformer.tools.benchmark
```
Outputs average forward-pass latency for SpiralFormer vs. a vanilla Transformer on synthetic glyph data (CPU).

## Status

Code is **prototype-level**: ideal for experimentation, not production.  Contributions welcome – open an issue or PR in the `docs/` folder first with your proposal.

## License

See top-level project LICENSE for multi-layered licensing (CC BY-SA 4.0 for theory, GPLv3 for software, etc.).
