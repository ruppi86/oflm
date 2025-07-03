
## 7 · CLI cheat–sheet  
(Existing raw commands kept for copy-paste convenience)

# cordyceps

## Small
python -m neural_mycelic_emulator.models.trainer  cordyceps_small  

python -m neural_mycelic_emulator.models.compare_stats           cordyceps_small           neural_mycelic_emulator/models/cordyceps_small/cordyceps_small_best.pt          neural_mycelic_emulator/dataset/Cordyceps_militari/Cordyceps_militari.txt

python -m neural_mycelic_emulator.models.evaluate_perplexity           cordyceps_small           neural_mycelic_emulator/models/cordyceps_small/cordyceps_small_best.pt           neural_mycelic_emulator/dataset/Cordyceps_militari/Cordyceps_militari.txt

## Medium

python -m neural_mycelic_emulator.models.trainer  cordyceps_medium      

python -m neural_mycelic_emulator.models.compare_stats           cordyceps_medium           neural_mycelic_emulator/models/cordyceps_medium/cordyceps_medium_best.pt          neural_mycelic_emulator/dataset/Cordyceps_militari/Cordyceps_militari.txt

python -m neural_mycelic_emulator.models.evaluate_perplexity           cordyceps_medium           neural_mycelic_emulator/models/cordyceps_medium/cordyceps_medium_best.pt           neural_mycelic_emulator/dataset/Cordyceps_militari/Cordyceps_militari.txt

 ## Large

python -m neural_mycelic_emulator.models.trainer       cordyceps_large       

python -m neural_mycelic_emulator.models.compare_stats cordyceps_large neural_mycelic_emulator/models/cordyceps_large/cordyceps_large_best.pt neural_mycelic_emulator/dataset/Cordyceps_militari/Cordyceps_militari.txt


# Enoki

## Small
python -m neural_mycelic_emulator.models.trainer enoki_small

python -m neural_mycelic_emulator.models.compare_stats enoki_small neural_mycelic_emulator/models/enoki_small/enoki_small_best.pt

python -m neural_mycelic_emulator.models.evaluate_perplexity enoki_small neural_mycelic_emulator/models/enoki_small/enoki_small_best.pt

## Medium

python -m neural_mycelic_emulator.models.trainer enoki_medium

python -m neural_mycelic_emulator.models.compare_stats enoki_medium neural_mycelic_emulator/models/enoki_medium/enoki_medium_best.pt


python -m neural_mycelic_emulator.models.evaluate_perplexity enoki_medium neural_mycelic_emulator/models/enoki_medium/enoki_medium_best.pt

## Large
python -m neural_mycelic_emulator.models.trainer enoki_large

python -m neural_mycelic_emulator.models.compare_stats enoki_large neural_mycelic_emulator/models/enoki_large/enoki_large_best.pt

python -m neural_mycelic_emulator.models.evaluate_perplexity enoki_large neural_mycelic_emulator/models/enoki_large/enoki_large_best.pt


# Ghost Fungi (Omphalotus nidiformis)

## Medium
python -m neural_mycelic_emulator.models.trainer ghost_medium

python -m neural_mycelic_emulator.models.compare_stats ghost_medium neural_mycelic_emulator/models/ghost_medium/ghost_medium_best.pt

python -m neural_mycelic_emulator.models.evaluate_perplexity ghost_medium neural_mycelic_emulator/models/ghost_medium/ghost_medium_best.pt

## Large
python -m neural_mycelic_emulator.models.trainer ghost_large

python -m neural_mycelic_emulator.models.compare_stats ghost_large neural_mycelic_emulator/models/ghost_large/ghost_large_best.pt

python -m neural_mycelic_emulator.models.evaluate_perplexity ghost_large neural_mycelic_emulator/models/ghost_large/ghost_large_best.pt


# Schizophyllum commune

## Small
python -m neural_mycelic_emulator.models.trainer schizo_small

python -m neural_mycelic_emulator.models.compare_stats schizo_small neural_mycelic_emulator/models/schizo_small/schizo_small_best.pt

python -m neural_mycelic_emulator.models.evaluate_perplexity schizo_small neural_mycelic_emulator/models/schizo_small/schizo_small_best.pt

## Medium
python -m neural_mycelic_emulator.models.trainer schizo_medium

python -m neural_mycelic_emulator.models.compare_stats schizo_medium neural_mycelic_emulator/models/schizo_medium/schizo_medium_best.pt

python -m neural_mycelic_emulator.models.evaluate_perplexity schizo_medium neural_mycelic_emulator/models/schizo_medium/schizo_medium_best.pt

## Large
python -m neural_mycelic_emulator.models.trainer schizo_large

python -m neural_mycelic_emulator.models.compare_stats schizo_large neural_mycelic_emulator/models/schizo_large/schizo_large_best.pt

python -m neural_mycelic_emulator.models.evaluate_perplexity schizo_large neural_mycelic_emulator/models/schizo_large/schizo_large_best.pt



