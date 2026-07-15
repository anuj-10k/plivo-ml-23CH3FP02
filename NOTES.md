# Notes

The model uses only audio samples before `pause_start`, plus the causal turn position and pause index known at decision time.  
Its acoustic signal combines energy level and decay, voiced-frame ratios, pitch distribution and slope, final voiced-island duration, zero-crossing behavior, and a rough syllable-rate proxy over four context windows.  
Training combines English and Hindi and keeps complete turns together in each validation fold.  
The final probability is a 4:1 soft blend of regularized Extra Trees and balanced logistic regression.  
Grouped validation improves English response delay from 1600 ms to 1078 ms at the same 5% interruption ceiling, while Hindi reaches 0.790 AUC but ties its 850 ms silence-only operating point.  
The hardest false positives are continuation pauses whose preceding speech has terminal-looking energy and pitch, such as `en__090` pause 1 and `hi__085` pause 1.  
The hardest misses are acoustically nonterminal endings, including `en__078` pause 0 and `hi__033` pause 2.  
With one more day I would collect speaker-normalized prosody across adjacent turns, tune directly with nested grouped validation for the constrained latency metric, and manually annotate the linguistic patterns in the worst errors.

