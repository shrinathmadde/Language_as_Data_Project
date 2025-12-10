Using device: cuda
======================================================================
DELIVERABLE 4: MODEL VARIANTS (4 CATEGORIES)
======================================================================

Loading Finnish...
Loading Hindi...

######################################################################
# FINNISH
######################################################################

Word vocabulary: 14091

============================================================
BASELINE: LSTM
============================================================
Parameters: 8,281,355
  Epoch  1 | Train PPL:   142.69 | Dev PPL:    83.53 | Time: 73.8s
  Epoch  2 | Train PPL:    47.61 | Dev PPL:   106.10 | Time: 73.6s
  Epoch  3 | Train PPL:    27.23 | Dev PPL:   144.34 | Time: 73.6s
  Epoch  4 | Train PPL:    19.36 | Dev PPL:   193.81 | Time: 73.4s
  Epoch  5 | Train PPL:    15.54 | Dev PPL:   222.40 | Time: 73.4s
  Epoch  6 | Train PPL:    13.91 | Dev PPL:   257.13 | Time: 73.4s
  Early stopping! Best Dev PPL: 83.53
Test PPL: 248.12

============================================================
CATEGORY 1 (Input): BPE TOKENIZATION
============================================================
BPE vocabulary: 5000
Parameters: 3,617,672
  Epoch  1 | Train PPL:   267.74 | Dev PPL:   220.44 | Time: 67.9s
  Epoch  2 | Train PPL:   112.92 | Dev PPL:   209.42 | Time: 67.9s
  Epoch  3 | Train PPL:    84.72 | Dev PPL:   216.01 | Time: 68.1s
  Epoch  4 | Train PPL:    71.12 | Dev PPL:   225.32 | Time: 68.0s
  Epoch  5 | Train PPL:    62.82 | Dev PPL:   234.40 | Time: 68.0s
  Epoch  6 | Train PPL:    55.65 | Dev PPL:   238.57 | Time: 68.0s
  Epoch  7 | Train PPL:    52.37 | Dev PPL:   243.13 | Time: 68.0s
  Early stopping! Best Dev PPL: 209.42
Test PPL: 246.15

============================================================
CATEGORY 2 (Architecture): LSTM + SELF-ATTENTION
============================================================
Parameters: 8,545,035
  Epoch  1 | Train PPL:    43.99 | Dev PPL:   150.49 | Time: 77.7s
  Epoch  2 | Train PPL:    11.34 | Dev PPL:   230.09 | Time: 77.7s
  Epoch  3 | Train PPL:     7.74 | Dev PPL:   291.73 | Time: 77.9s
  Epoch  4 | Train PPL:     6.18 | Dev PPL:   369.41 | Time: 77.9s
  Epoch  5 | Train PPL:     4.89 | Dev PPL:   464.80 | Time: 77.9s
  Epoch  6 | Train PPL:     4.43 | Dev PPL:   534.18 | Time: 77.9s
  Early stopping! Best Dev PPL: 150.49
Test PPL: 548.90

============================================================
CATEGORY 3 (Training): LABEL SMOOTHING (0.1)
============================================================
Parameters: 8,281,355
  Epoch  1 | Train PPL:   265.09 | Dev PPL:   164.57 | Time: 77.1s
  Epoch  2 | Train PPL:   116.49 | Dev PPL:   181.08 | Time: 76.9s
  Epoch  3 | Train PPL:    76.03 | Dev PPL:   202.23 | Time: 77.0s
  Epoch  4 | Train PPL:    57.89 | Dev PPL:   217.12 | Time: 77.0s
  Epoch  5 | Train PPL:    48.23 | Dev PPL:   225.72 | Time: 77.0s
  Epoch  6 | Train PPL:    44.00 | Dev PPL:   234.78 | Time: 77.0s
  Early stopping! Best Dev PPL: 164.57
Test PPL: 102.59

============================================================
CATEGORY 4 (Sampling): DIFFERENT DECODING STRATEGIES
============================================================

Prompt: 'hän on'
--------------------------------------------------
Greedy:       hän on <UNK> <UNK> <UNK> , <UNK> <UNK> ja <UNK> <UNK> <UNK> .
Temp=0.7:     hän on toiminut professorina <UNK> perustamista sijaitsevasta <UNK> <UNK> <UNK> <UNK> .
Temp=1.5:     hän on kaksi lisää melkein <UNK> <UNK> .
Top-k (k=10): hän on julkaissut the eraser <UNK> , <UNK> <UNK> , joka oli keskittynyt <UNK> ja <UNK> .
Top-p (p=0.9): hän on näytellyt <UNK> sdp:n tukemiseen .

Prompt: 'se oli'
--------------------------------------------------
Greedy:       se oli <UNK> <UNK> , mutta <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> .
Temp=0.7:     se oli 1800-luvun <UNK> .
Temp=1.5:     se oli kuulemma pitkä vaikeaa hän ei vähän kello .
Top-k (k=10): se oli veljeni , mutta <UNK> työllähän siitäkin selvitään , jotka on myös <UNK> <UNK> ohjeet .
Top-p (p=0.9): se oli paljon <UNK> , mutta <UNK> olen varma , että otin <UNK> hänen <UNK> <UNK> <UNK> , kunnes päätettiin mielessä , että kuningatar <UNK> rakennetaan <UNK> <UNK> , missä tuli hieman

######################################################################
# HINDI
######################################################################

Word vocabulary: 9631

============================================================
BASELINE: LSTM
============================================================
Parameters: 5,993,375
  Epoch  1 | Train PPL:    90.90 | Dev PPL:    75.46 | Time: 95.1s
  Epoch  2 | Train PPL:    36.97 | Dev PPL:    82.88 | Time: 95.2s
  Epoch  3 | Train PPL:    26.33 | Dev PPL:    96.40 | Time: 95.3s
  Epoch  4 | Train PPL:    21.51 | Dev PPL:   110.17 | Time: 95.2s
  Epoch  5 | Train PPL:    18.62 | Dev PPL:   118.29 | Time: 95.1s
  Epoch  6 | Train PPL:    17.32 | Dev PPL:   126.17 | Time: 95.1s
  Early stopping! Best Dev PPL: 75.46
Test PPL: 118.78

============================================================
CATEGORY 1 (Input): BPE TOKENIZATION
============================================================
BPE vocabulary: 5000
Parameters: 3,617,672
  Epoch  1 | Train PPL:   112.11 | Dev PPL:    99.19 | Time: 75.4s
  Epoch  2 | Train PPL:    52.51 | Dev PPL:    92.63 | Time: 75.9s
  Epoch  3 | Train PPL:    40.34 | Dev PPL:    94.40 | Time: 75.6s
  Epoch  4 | Train PPL:    34.32 | Dev PPL:    99.00 | Time: 75.5s
  Epoch  5 | Train PPL:    30.57 | Dev PPL:   103.85 | Time: 75.5s
  Epoch  6 | Train PPL:    27.53 | Dev PPL:   105.72 | Time: 75.5s
  Epoch  7 | Train PPL:    26.10 | Dev PPL:   108.26 | Time: 75.5s
  Early stopping! Best Dev PPL: 92.63
Test PPL: 106.87

============================================================
CATEGORY 2 (Architecture): LSTM + SELF-ATTENTION
============================================================
Parameters: 6,257,055
  Epoch  1 | Train PPL:    42.20 | Dev PPL:    86.72 | Time: 101.5s
  Epoch  2 | Train PPL:    17.73 | Dev PPL:   102.21 | Time: 101.3s
  Epoch  3 | Train PPL:    13.65 | Dev PPL:   111.80 | Time: 101.4s
  Epoch  4 | Train PPL:    11.68 | Dev PPL:   122.53 | Time: 101.3s
  Epoch  5 | Train PPL:     9.71 | Dev PPL:   136.69 | Time: 101.4s
  Epoch  6 | Train PPL:     9.04 | Dev PPL:   150.45 | Time: 101.4s
  Early stopping! Best Dev PPL: 86.72
Test PPL: 136.18

============================================================
CATEGORY 3 (Training): LABEL SMOOTHING (0.1)
============================================================
Parameters: 5,993,375
  Epoch  1 | Train PPL:   182.37 | Dev PPL:   147.40 | Time: 101.2s
  Epoch  2 | Train PPL:    92.47 | Dev PPL:   143.13 | Time: 101.3s
  Epoch  3 | Train PPL:    71.98 | Dev PPL:   146.83 | Time: 101.3s
  Epoch  4 | Train PPL:    61.61 | Dev PPL:   151.68 | Time: 101.4s
  Epoch  5 | Train PPL:    55.10 | Dev PPL:   155.66 | Time: 101.3s
  Epoch  6 | Train PPL:    50.01 | Dev PPL:   156.73 | Time: 101.3s
  Epoch  7 | Train PPL:    47.68 | Dev PPL:   159.52 | Time: 101.3s
  Early stopping! Best Dev PPL: 143.13
Test PPL: 66.62

============================================================
CATEGORY 4 (Sampling): DIFFERENT DECODING STRATEGIES
============================================================

Prompt: 'वह है'
--------------------------------------------------
Greedy:       वह है कि इस तरह की प्रताड़ना से पहले ही इस मुद्दे पर बातचीत करेंगे ।
Temp=0.7:     वह है , इसलिए उन्होंने खुद को असहज नहीं ली ।
Temp=1.5:     वह है ।
Top-k (k=10): वह है कि वह युद्धक <UNK> में <UNK> और <UNK> का भूजा चाहिए ।
Top-p (p=0.9): वह है कि यह मुलाक़ात अपने निजी क्षेत्र के अनुरूप हो जा रहा है ।

Prompt: 'यह एक'
--------------------------------------------------
Greedy:       यह एक अनूठा संग्रहालय है ।
Temp=0.7:     यह एक अन्य अधिकारी को गिरफ्तार किया गया है ।
Temp=1.5:     यह एक क्षेत्रीय रजिस्टर माना गया है ।
Top-k (k=10): यह एक ऐतिहासिक <UNK> है ।
Top-p (p=0.9): यह एक नया और अध्ययन है , लेकिन यदि , मेहर और ट्रेन में ऐसी चमक की वजह से कोई ज्यादा नहीं है ।

======================================================================
SUMMARY: ALL VARIANTS
======================================================================

Variant                   Category             Finnish PPL     Hindi PPL      
---------------------------------------------------------------------------
baseline                  Baseline             248.12          118.78         
bpe                       1. Input Rep         246.15          106.87         
attention                 2. Architecture      548.90          136.18         
label_smoothing           3. Training          102.59          66.62          

Plot saved: deliverable4_comparison.png
Results saved: deliverable4_results.json
