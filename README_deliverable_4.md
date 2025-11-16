(venv) shrinath@dellpro:~/ShriCode/Langugae-as-Data/src$ python deliverable_4_tokenizer_analysis.py
================================================================================
DELIVERABLE 4: MORPHOLOGICAL ANALYSIS WITH SUB-WORD TOKENIZERS
================================================================================

================================================================================
FINNISH TOKENIZER
================================================================================

Loading data from ../data/ud/UD_Finnish-TDT/fi_tdt-ud-train.conllu...
Loaded 162815 words from 12217 sentences

Loading data from ../data/ud/UD_Finnish-TDT/fi_tdt-ud-test.conllu...
Loaded 21070 words from 1555 sentences

============================================================
Creating BPE Tokenizer for Finnish
============================================================
Vocabulary size: 5000
Normalization settings:
  - NFKC normalization: True
  - Strip accents: False
  - Lowercase: False

Training BPE tokenizer...
[00:00:00] Pre-processing files (1 Mo)    ████████████████████                100%[00:00:00] Tokenize words                 ████████████████████ 48108    /    48108
[00:00:00] Count pairs                    ████████████████████ 48108    /    48108
[00:00:00] Compute merges                 ████████████████████ 4755     /     4755
Training complete!
Tokenizer saved to: tokenizer_Finnish_BPE.json

============================================================
Analyzing Tokenization for Finnish
============================================================

Word categories in training data:
  Frequent words (>=100): 98
  Rare words (2-5): 11217
  Hapax (frequency=1): 34753
  Unseen in test: 4372

Category        Tokens/Word     Avg Token Len  
---------------------------------------------
frequent        1.00            4.07           
rare            2.38            3.64           
hapax           3.34            3.35           
unseen          3.54            3.08           

Analyzing full test corpus...

Overall test corpus statistics:
  Total tokens generated: 23897
  Unique tokens: 3469
  Average tokens per word: 1.77
  Std tokens per word: 1.13

Top 20 most frequent tokens:
   1. .                        849
   2. ,                        815
   3. ja                       495
   4. on                       322
   5. että                     158
   6. se                       148
   7. -                        131
   8. a                        130
   9. ei                       129
  10. t                        101
  11. )                         96
  12. ta                        94
  13. i                         91
  14. si                        87
  15. (                         85
  16. kin                       79
  17. e                         78
  18. lle                       77
  19. en                        76
  20. oli                       75

============================================================
Morphological Segmentation Analysis for Finnish
============================================================

Plurals (-t):
  talo                 → talo
  talot                → talo | t
  kissa                → kissa
  kissat               → kissa | t
  koira                → koi | ra
  koirat               → koi | rat

Case markers:
  talo                 → talo
  talossa              → talo | ssa
  talosta              → talo | sta
  taloon               → talo | on
  talon                → talon

Possessive suffixes:
  talo                 → talo
  taloni               → tal | oni
  talosi               → talo | si
  talonsa              → talon | sa

Compounds:
  kirjasto             → kirja | sto
  kirja                → kirja
  talo                 → talo
  kahvikuppi           → kah | vi | ku | ppi
  kahvi                → kah | vi
  kuppi                → ku | ppi

============================================================
Segmentation Consistency Analysis
============================================================

Words ending in '-t' (potential plurals):
  meidät               → mei | dät
  ihmiset              → ihmiset
  uskaltautuivat       → uskalta | utuivat
  onnistunut           → onnistunut
  noussut              → noussut
  onnistunut           → onnistunut
  tapahtumat           → tapahtu | mat
  alkoivat             → alkoi | vat
  vienyt               → vien | yt
  tullut               → tullut

Visualization saved: tokenizer_analysis_Finnish_BPE.png

================================================================================
HINDI TOKENIZER
================================================================================

Loading data from ../data/ud/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu...
Loaded 281057 words from 13306 sentences

Loading data from ../data/ud/UD_Hindi-HDTB/hi_hdtb-ud-test.conllu...
Loaded 35430 words from 1684 sentences

============================================================
Creating BPE Tokenizer for Hindi
============================================================
Vocabulary size: 5000
Normalization settings:
  - NFKC normalization: True
  - Strip accents: False
  - Lowercase: False

Training BPE tokenizer...
[00:00:00] Pre-processing files (3 Mo)    ████████████████████                100%[00:00:00] Tokenize words                 ████████████████████ 16651    /    16651
[00:00:00] Count pairs                    ████████████████████ 16651    /    16651
[00:00:00] Compute merges                 ████████████████████ 4896     /     4896
Training complete!
Tokenizer saved to: tokenizer_Hindi_BPE.json

============================================================
Analyzing Tokenization for Hindi
============================================================

Word categories in training data:
  Frequent words (>=100): 338
  Rare words (2-5): 5362
  Hapax (frequency=1): 7252
  Unseen in test: 1154

Category        Tokens/Word     Avg Token Len  
---------------------------------------------
frequent        1.00            3.38           
rare            2.42            2.34           
hapax           2.83            2.26           
unseen          2.91            2.17           

Analyzing full test corpus...

Overall test corpus statistics:
  Total tokens generated: 25629
  Unique tokens: 3000
  Average tokens per word: 1.20
  Std tokens per word: 0.58

Top 20 most frequent tokens:
   1. के                     1,037
   2. ।                        992
   3. में                      635
   4. की                       602
   5. है                       565
   6. को                       476
   7. ने                       404
   8. कि                       375
   9. से                       349
  10. का                       312
  11. पर                       257
  12. और                       255
  13. ,                        201
  14. कहा                      201
  15. इस                       177
  16. हैं                      170
  17. भी                       143
  18. -                        141
  19. कर                       137
  20. लिए                      118

============================================================
Morphological Segmentation Analysis for Hindi
============================================================

Plurals:
  लड़का                → लड़ | का
  लड़के                → लड़ | के
  लड़की                → लड़की
  लड़कियाँ             → लड़ | किया | ँ

Case markers:
  घर                   → घर
  घर में               → घर | में
  घर से                → घर | से
  घर को                → घर | को

Verb forms:
  जाना                 → जाना
  जाता                 → जाता
  जाती                 → जाती
  गया                  → गया
  गई                   → गई

============================================================
Segmentation Consistency Analysis
============================================================

Visualization saved: tokenizer_analysis_Hindi_BPE.png

================================================================================
ANALYSIS COMPLETE!
================================================================================

Generated files:
  - tokenizer_Finnish_BPE.json
  - tokenizer_Hindi_BPE.json
  - tokenizer_analysis_Finnish_BPE.png
  - tokenizer_analysis_Hindi_BPE.png
(venv) shrinath@dellpro:~/ShriCode/Langugae-as-Data/src$