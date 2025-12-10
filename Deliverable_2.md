(venv) shrinath@dellpro:~/ShriCode/Langugae-as-Data$ python Deliverable_2.py

============================================================
N-GRAM LANGUAGE MODEL: FINNISH
============================================================

----------------------------------------
2-gram Model
----------------------------------------
Train: 12217 sentences
Dev: 1364 sentences
Test: 1555 sentences
Vocabulary size: 14090
Words below threshold: 32206
Training 2-gram model on 12217 sentences...
Unique 2-grams: 83117
Unique contexts: 14089

Results:
  Dev  Perplexity: 256.20 (OOV: 29.77%)
  Test Perplexity: 266.52 (OOV: 28.61%)

----------------------------------------
3-gram Model
----------------------------------------
Train: 12217 sentences
Dev: 1364 sentences
Test: 1555 sentences
Vocabulary size: 14090
Words below threshold: 32206
Training 3-gram model on 12217 sentences...
Unique 3-grams: 128564
Unique contexts: 82600

Results:
  Dev  Perplexity: 1208.95 (OOV: 29.77%)
  Test Perplexity: 1257.93 (OOV: 28.61%)

----------------------------------------
4-gram Model
----------------------------------------
Train: 12217 sentences
Dev: 1364 sentences
Test: 1555 sentences
Vocabulary size: 14090
Words below threshold: 32206
Training 4-gram model on 12217 sentences...
Unique 4-grams: 148650
Unique contexts: 123785

Results:
  Dev  Perplexity: 3379.54 (OOV: 29.77%)
  Test Perplexity: 3401.37 (OOV: 28.61%)

============================================================
N-GRAM LANGUAGE MODEL: HINDI
============================================================

----------------------------------------
2-gram Model
----------------------------------------
Train: 13306 sentences
Dev: 1659 sentences
Test: 1684 sentences
Vocabulary size: 9630
Words below threshold: 7252
Training 2-gram model on 13306 sentences...
Unique 2-grams: 101258
Unique contexts: 9629

Results:
  Dev  Perplexity: 202.55 (OOV: 6.20%)
  Test Perplexity: 192.36 (OOV: 6.30%)

----------------------------------------
3-gram Model
----------------------------------------
Train: 13306 sentences
Dev: 1659 sentences
Test: 1684 sentences
Vocabulary size: 9630
Words below threshold: 7252
Training 3-gram model on 13306 sentences...
Unique 3-grams: 195966
Unique contexts: 101226

Results:
  Dev  Perplexity: 921.92 (OOV: 6.20%)
  Test Perplexity: 861.14 (OOV: 6.30%)

----------------------------------------
4-gram Model
----------------------------------------
Train: 13306 sentences
Dev: 1659 sentences
Test: 1684 sentences
Vocabulary size: 9630
Words below threshold: 7252
Training 4-gram model on 13306 sentences...
Unique 4-grams: 241895
Unique contexts: 195350

Results:
  Dev  Perplexity: 2986.78 (OOV: 6.20%)
  Test Perplexity: 2834.04 (OOV: 6.30%)

============================================================
SUMMARY
============================================================

Model      Finnish Dev     Finnish Test    Hindi Dev       Hindi Test     
----------------------------------------------------------------------
2-gram     256.20          266.52          202.55          192.36         
3-gram     1208.95         1257.93         921.92          861.14         
4-gram     3379.54         3401.37         2986.78         2834.04