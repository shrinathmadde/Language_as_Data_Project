(venv) shrinath@dellpro:~/ShriCode/Langugae-as-Data/src$ python deliverable_3_corpus_statistics.py

Loading Finnish corpus from ../data/corpo/wordlist_fin_news_2012_300K_20251115235448.csv...
Loaded 394130 unique word forms
Loading sentences from ../data/ud/UD_Finnish-TDT/fi_tdt-ud-train.conllu...
Loaded 12217 sentences with 162815 tokens

Loading Hindi corpus from ../data/corpo/wordlist_hin_news_2019_20251112220411.csv...
Loaded 39213 unique word forms
Loading sentences from ../data/ud/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu...
Loaded 13306 sentences with 281057 tokens

================================================================================
BASIC STATISTICS
================================================================================

============================================================
Statistics for Finnish Corpus
============================================================
num_types............................... 394,130
num_tokens.............................. 3,784,991
type_token_ratio........................ 0.10
hapax_legomena.......................... 239,826
hapax_percentage........................ 60.85
num_sentences........................... 12,217
avg_sentence_length_words............... 13.33
std_sentence_length_words............... 9.49
p10_sentence_length_words............... 5.00
p90_sentence_length_words............... 23.00
avg_sentence_length_chars............... 86.42
std_sentence_length_chars............... 66.35
avg_word_length......................... 6.48
std_word_length......................... 4.29
p10_word_length......................... 1.00
p90_word_length......................... 12.00

============================================================
Statistics for Hindi Corpus
============================================================
num_types............................... 39,213
num_tokens.............................. 10,584,904
type_token_ratio........................ 0.00
hapax_legomena.......................... 0
hapax_percentage........................ 0.00
num_sentences........................... 13,306
avg_sentence_length_words............... 21.12
std_sentence_length_words............... 9.50
p10_sentence_length_words............... 11.00
p90_sentence_length_words............... 34.00
avg_sentence_length_chars............... 80.91
std_sentence_length_chars............... 39.10
avg_word_length......................... 3.83
std_word_length......................... 2.16
p10_word_length......................... 2.00
p90_word_length......................... 7.00

================================================================================
TYPE-TOKEN RATIO COMPARISON
================================================================================
Finnish TTR: 0.104130
Hindi TTR:   0.003705
Difference:  0.100425

================================================================================
TOP 20 MOST FREQUENT WORDS
================================================================================

Finnish:
 1. .                  283,489
 2. ,                  162,767
 3. on                  93,207
 4. ja                  85,619
 5. että                29,424
 6. ei                  28,556
 7. oli                 19,784
 8. mukaan              19,399
 9. myös                17,136
10. ovat                16,079
11. mutta               14,170
12. "                   13,484
13. se                  12,490
14. ole                 11,939
15. kun                 11,440
16. hän                 10,016
17. jo                   9,036
18. sen                  8,261
19. kuin                 8,211
20. viime                8,092

Hindi:
 1. के                 417,400
 2. .                  375,142
 3. में                324,300
 4. की                 253,645
 5. है                 240,716
 6. को                 201,336
 7. से                 175,324
 8. ,                  172,730
 9. ने                 172,108
10. और                 157,998
11. का                 134,690
12. कि                 129,153
13. पर                 123,094
14. हैं                100,098
15. भी                  86,699
16. इस                  70,020
17. नहीं                69,110
18. कहा                 60,837
19. एक                  60,825
20. लिए                 59,327

================================================================================
BIGRAM ANALYSIS
================================================================================

Finnish - Top 15 Bigrams:
 1. , että                           1,076
 2. , mutta                            435
 3. , ja                               434
 4. , joka                             335
 5. ei ole                             224
 6. , kun                              201
 7. , jotka                            192
 8. , jonka                            160
 9. , jossa                            156
10. , jos                              137
11. , sillä                            119
12. siitä ,                            113
13. se on                              105
14. että ei                             98
15. , koska                             94

Hindi - Top 15 Bigrams:
 1. है ।                             4,283
 2. कहा कि                           1,713
 3. के लिए                           1,639
 4. हैं ।                            1,450
 5. है कि                            1,272
 6. ने कहा                             913
 7. उन्होंने कहा                       636
 8. के बाद                             597
 9. था ।                               589
10. गया है                             555
11. के साथ                             531
12. बताया कि                           467
13. थी ।                               397
14. रहा है                             387
15. थे ।                               385

================================================================================
TRIGRAM ANALYSIS
================================================================================

Finnish - Top 10 Trigrams:
 1. , että ei                                     74
 2. siitä , että                                  61
 3. , joka on                                     55
 4. ( EY )                                        55
 5. EY ) N:o                                      55
 6. Arvoisa puhemies ,                            44
 7. se , että                                     41
 8. , ottaa huomioon                              27
 9. , joka oli                                    26
10. sitä , että                                   26

Hindi - Top 10 Trigrams:
 1. ने कहा कि                                    779
 2. उन्होंने कहा कि                              597
 3. गया है ।                                     377
 4. रहा है ।                                     273
 5. ने बताया कि                                  272
 6. नहीं है ।                                    243
 7. रहे हैं ।                                    222
 8. के बारे में                                  221
 9. कहना है कि                                   212
10. रही है ।                                     212

================================================================================
CLOSED WORD CLASS ANALYSIS: CONJUNCTIONS
================================================================================

Finnish Conjunctions:
  ja             :     85,619 (2.2621%)
  tai            :      6,462 (0.1707%)
  mutta          :     14,170 (0.3744%)
  että           :     29,424 (0.7774%)
  kun            :     11,440 (0.3022%)

Hindi Conjunctions:
  और             :    157,998 (1.4927%)
  या             :     10,762 (0.1017%)
  लेकिन          :     27,861 (0.2632%)
  परन्तु         :         54 (0.0005%)
  किन्तु         :         18 (0.0002%)

================================================================================
GENERATING VISUALIZATIONS
================================================================================

Saved: comparison_zipf_and_frequency.png
Saved: comparison_word_length_distribution.png
Saved: comparison_sentence_length_distribution.png

================================================================================
ANALYSIS COMPLETE!
================================================================================
(venv) shrinath@dellpro:~/ShriCode/Langugae-as-Data/src$