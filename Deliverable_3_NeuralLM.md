Using device: cuda
Configuration:
  embed_dim: 256
  hidden_dim: 256
  num_layers: 2
  dropout: 0.3
  seq_length: 30
  batch_size: 64
  learning_rate: 0.001
  epochs: 50
  patience: 5
  min_freq: 2

======================================================================
NEURAL LANGUAGE MODEL: FINNISH
======================================================================

Loading data...
Train: 12217 sentences
Dev: 1364 sentences
Test: 1555 sentences

Building vocabulary...
Vocabulary size: 14091
Words below threshold: 32206

Creating datasets...
Train sequences: 187219
Dev sequences: 21006
Test sequences: 24150

Creating model...
Total parameters: 4,674,059
Trainable parameters: 4,674,059

Training for up to 50 epochs with patience=5
------------------------------------------------------------
Epoch   1 | Train Loss: 5.1406 | Train PPL:   170.82 | Dev PPL:    75.64 | LR: 0.001000 | Time: 70.1s
Epoch   2 | Train Loss: 4.3431 | Train PPL:    76.94 | Dev PPL:    77.63 | LR: 0.001000 | Time: 70.3s
Epoch   3 | Train Loss: 3.8666 | Train PPL:    47.78 | Dev PPL:    85.73 | LR: 0.001000 | Time: 70.8s
Epoch   4 | Train Loss: 3.4986 | Train PPL:    33.07 | Dev PPL:    95.90 | LR: 0.000500 | Time: 70.8s
Epoch   5 | Train Loss: 3.2066 | Train PPL:    24.69 | Dev PPL:   102.53 | LR: 0.000500 | Time: 70.8s
Epoch   6 | Train Loss: 3.0518 | Train PPL:    21.15 | Dev PPL:   109.70 | LR: 0.000500 | Time: 70.6s

Early stopping at epoch 6! Best Dev PPL: 75.64

Evaluating on test set...
Test Perplexity: 114.42
Learning curves saved to: learning_curves_Finnish.png
Model saved to: model_Finnish.pt

======================================================================
NEURAL LANGUAGE MODEL: HINDI
======================================================================

Loading data...
Train: 13306 sentences
Dev: 1659 sentences
Test: 1684 sentences

Building vocabulary...
Vocabulary size: 9631
Words below threshold: 7252

Creating datasets...
Train sequences: 307639
Dev sequences: 38505
Test sequences: 38768

Creating model...
Total parameters: 3,527,839
Trainable parameters: 3,527,839

Training for up to 50 epochs with patience=5
------------------------------------------------------------
Epoch   1 | Train Loss: 4.7590 | Train PPL:   116.63 | Dev PPL:    71.61 | LR: 0.001000 | Time: 91.6s
Epoch   2 | Train Loss: 4.0080 | Train PPL:    55.04 | Dev PPL:    66.11 | LR: 0.001000 | Time: 91.5s
Epoch   3 | Train Loss: 3.6718 | Train PPL:    39.32 | Dev PPL:    65.89 | LR: 0.001000 | Time: 91.5s
Epoch   4 | Train Loss: 3.4384 | Train PPL:    31.14 | Dev PPL:    67.49 | LR: 0.001000 | Time: 91.6s
Epoch   5 | Train Loss: 3.2612 | Train PPL:    26.08 | Dev PPL:    69.93 | LR: 0.001000 | Time: 91.6s
Epoch   6 | Train Loss: 3.1211 | Train PPL:    22.67 | Dev PPL:    71.70 | LR: 0.000500 | Time: 91.8s
Epoch   7 | Train Loss: 2.9859 | Train PPL:    19.80 | Dev PPL:    73.95 | LR: 0.000500 | Time: 91.6s
Epoch   8 | Train Loss: 2.9192 | Train PPL:    18.53 | Dev PPL:    75.59 | LR: 0.000500 | Time: 91.6s

Early stopping at epoch 8! Best Dev PPL: 65.89

Evaluating on test set...
Test Perplexity: 70.87
Learning curves saved to: learning_curves_Hindi.png
Model saved to: model_Hindi.pt

======================================================================
SUMMARY: NEURAL LANGUAGE MODEL RESULTS
======================================================================

Metric                    Finnish         Hindi          
-------------------------------------------------------
Vocabulary Size           14,091          9,631          
Epochs Trained            6               8              
Final Train PPL           21.15           18.53          
Best Dev PPL              75.64           65.89          
Test PPL                  114.42          70.87          

======================================================================
COMPARISON WITH N-GRAM BASELINE
======================================================================

Model                Finnish Test PPL     Hindi Test PPL      
------------------------------------------------------------
2-gram               266.52               192.36              
3-gram               1257.93              861.14              
4-gram               3401.37              2834.04             
Neural LM (LSTM)     114.42               70.87               

Improvement over best n-gram (2-gram):
  Finnish: 57.1%
  Hindi: 63.2%

Results saved to: neural_lm_results.json
