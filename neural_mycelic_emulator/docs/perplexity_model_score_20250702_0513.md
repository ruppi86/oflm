

python -m neural_mycelic_emulator.models.evaluate_perplexity      

     cordyceps_small           neural_mycelic_emulator/models/cordyceps_small/cordyceps_small_best.pt           neur
al_mycelic_emulator/dataset/Cordyceps_militari/Cordyceps_militari.txt   

Perplexity: 4.945

python -m neural_mycelic_emulator.models.compare_stats           cordyceps_small           neural_mycelic_emulator/models/cordyceps_small/cordyceps_small_best.pt          neural_mycelic_emulator/dataset/Cordyceps_militari/Cordyceps_militari.txt    

Silence ratio real  : 0.00
Silence ratio synth : 0.00  Δ=0.00
ISI KS-stat=0.001  p=1.000
Glyph freq L1-diff  : 0.577

python -m neural_mycelic_emulator.models.evaluate_perplexity           cordyceps_medium           neural_mycelic_emulator/models/cordyceps_medium/cordyceps_medium_best.pt           n
eural_mycelic_emulator/dataset/Cordyceps_militari/Cordyceps_militari.txt

Perplexity: 3.627

python -m neural_mycelic_emulator.models.compare_stats           cordyceps_medium           neural_mycelic_emulator/models/cordyceps_medium/cordyceps_medium_best.pt          neural_m
ycelic_emulator/dataset/Cordyceps_militari/Cordyceps_militari.txt

Silence ratio real  : 0.00
Silence ratio synth : 0.00  Δ=0.00
ISI KS-stat=0.000  p=1.000
Glyph freq L1-diff  : 0.244