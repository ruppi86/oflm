# 🌱 haikumeadowlib — Training Log
*A living record of breath, silence, and emergence.*

---

## 📅 Training session metadata

- **Run date:** YYYY-MM-DD
- **Model version:** haikumeadowlib v0.x
- **Device:** (e.g. CPU, GPU, system specs)
- **RAM used:** (approx. GB)
- **Time elapsed:** (e.g. 7h 32m)
- **Script:** `train_haikumeadow.py`

---

## 📜 Corpus used

| Source                              | Count  | Notes                            |
|-------------------------------------|--------|----------------------------------|
| Classical haiku (J/E)               | XXXX   | Public domain, preprocessed      |
| Micro-poetry fragments (< 40 chars) | XXXX   | Filtered, downsampled            |
| Silence tokens (`…`, em-dash, etc)  | ✓      | Included                         |
| Seasonal vector mappings            | ✓      | External JSON, fixed per epoch   |

**Final size after distillation:** XX,XXX lines  
**Retention after compost mask:** XX–XX %

---

## ⚙️ Training configuration

```json
{
  "epochs": 5,
  "batch_size": 32,
  "optimizer": "Adam",
  "learning_rate": 0.001,
  "scheduler": "ExponentialDecay",
  "loss_fn": "CrossEntropy",
  "embedding_dim": 128,
  "hidden_dim": 256,
  "rnn_type": "GRU",
  "dropout": 0.1,
  "season_vector_dim": 8
}
```

---

## 🌬 Pulmonos pacing

- **Breath rate:** 0.5 Hz (1 inhale + 1 exhale every 2 sec)
- **Curriculum timing:** batches released on exhale ticks only

---

## 🌀 Notes & reflections

> "The model began to prefer silence after the second epoch.  
Its output shifted from brightness to mist."

- [ ] Was the decay-mask effective?
- [ ] Did the rhythm of the training feel natural?
- [ ] Any surprises in loss curve or output?
- [ ] Will you reuse this model for solstice fine-tuning?

---

## 📦 Artifacts

- `haikumeadowlib_v0.1.pt`
- `train_config.json`
- `corpus_used.txt`

---

*“When the breath forgets the word, the meadow speaks.”*
