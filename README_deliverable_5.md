(venv) shrinath@dellpro:~/ShriCode/Langugae-as-Data$ python src/deliverable_5_dependency_parsing.py
================================================================================
DELIVERABLE 5: DEPENDENCY PARSING AND SYNTACTIC ANALYSIS
================================================================================

================================================================================
Finnish DEPENDENCY PARSING
================================================================================

Loading SpaCy model: fi_core_news_sm
Model loaded successfully!
Pipeline components: ['tok2vec', 'tagger', 'morphologizer', 'parser', 'lemmatizer', 'attribute_ruler', 'ner']

Loading gold standard from data/ud/UD_Finnish-TDT/fi_tdt-ud-test.conllu...
Loaded 500 gold standard trees

Parsing 500 sentences...
  Parsed 100/500 sentences...
  Parsed 200/500 sentences...
  Parsed 300/500 sentences...
  Parsed 400/500 sentences...
  Parsed 500/500 sentences...
Parsing complete!

Computing tree statistics for 500 sentences...

============================================================
Dependency Tree Statistics - Finnish
============================================================

Tree Depth:
  Average: 3.25
  Std Dev: 1.79
  Min: 0
  Max: 13

Node Degree Distribution:
  Degree 0:  4,236 nodes (64.66%)
  Degree 1:    926 nodes (14.14%)
  Degree 2:    391 nodes ( 5.97%)
  Degree 3:    310 nodes ( 4.73%)
  Degree 4:    319 nodes ( 4.87%)
  Degree 5:    187 nodes ( 2.85%)

Average Distance to Root by POS:
  X           : 0.00
  INTJ        : 1.30
  VERB        : 1.46
  AUX         : 1.91
  PUNCT       : 1.98
  PROPN       : 2.14
  NOUN        : 2.17
  SYM         : 2.36
  ADV         : 2.40
  PRON        : 2.54

Most Common Leaf Node POS Tags:
  PUNCT       :  1,033 (24.39%)
  ADV         :    571 (13.48%)
  PRON        :    553 (13.05%)
  NOUN        :    551 (13.01%)
  AUX         :    476 (11.24%)
  CCONJ       :    341 ( 8.05%)
  ADJ         :    258 ( 6.09%)
  SCONJ       :    181 ( 4.27%)
  PROPN       :    105 ( 2.48%)
  ADP         :     70 ( 1.65%)

Most Common Ancestors (by child POS):
  NOUN:
    → VERB        :  1,937
    → NOUN        :    768
    → ADJ         :    190
    → PRON        :    176
    → ADV         :     88
  VERB:
    → VERB        :    856
    → NOUN        :    279
    → PRON        :     94
    → ADJ         :     85
    → ADV         :     41
  ADJ:
    → VERB        :    500
    → NOUN        :    450
    → ADJ         :     78
    → PRON        :     55
    → ADV         :     26

Most Common Descendants (by parent POS):
  NOUN:
    → NOUN        :    373
    → ADJ         :    277
    → PRON        :    220
    → PUNCT       :    217
    → VERB        :    155
  VERB:
    → NOUN        :    814
    → PUNCT       :    610
    → VERB        :    384
    → ADV         :    354
    → PRON        :    341
  ADJ:
    → PUNCT       :     89
    → NOUN        :     82
    → ADV         :     75
    → AUX         :     73
    → VERB        :     41

Most Common Dependency Relations:
  punct               :  1,027 (15.68%)
  advmod              :    623 ( 9.51%)
  obl                 :    529 ( 8.08%)
  ROOT                :    526 ( 8.03%)
  conj                :    410 ( 6.26%)
  obj                 :    395 ( 6.03%)
  nsubj               :    387 ( 5.91%)
  cc                  :    345 ( 5.27%)
  amod                :    286 ( 4.37%)
  aux                 :    251 ( 3.83%)
  nmod:poss           :    207 ( 3.16%)
  cop                 :    203 ( 3.10%)
  mark                :    190 ( 2.90%)
  nsubj:cop           :    182 ( 2.78%)
  det                 :    134 ( 2.05%)

============================================================
Example Dependency Parses - Finnish
============================================================

Example 1: Taas teatteriin
------------------------------------------------------------
Token           POS      Head            Dep Rel         Children
------------------------------------------------------------
Taas            ADV      teatteriin      advmod          
teatteriin      VERB     ROOT            ROOT            Taas

Example 2: Tänäänkin pitäisi mennä teatteriin .
------------------------------------------------------------
Token           POS      Head            Dep Rel         Children
------------------------------------------------------------
Tänäänkin       ADV      mennä           advmod          
pitäisi         AUX      mennä           aux             
mennä           VERB     ROOT            ROOT            Tänäänkin, pitäisi, teatteriin, .
teatteriin      NOUN     mennä           obl             
.               PUNCT    mennä           punct           

Example 3: Varasin pupulle ja minulle sekä sille sisarentyttärelleni , joka pääsi Turkuun lakia lukemaan , liput kaupunginteatterin Laulavat sadepisarat -musikaaliin .
------------------------------------------------------------
Token           POS      Head            Dep Rel         Children
------------------------------------------------------------
Varasin         VERB     ROOT            ROOT            pupulle, sille, sisarentyttärelleni
pupulle         NOUN     Varasin         obl             minulle
ja              CCONJ    minulle         cc              
minulle         PRON     pupulle         conj            ja
sekä            CCONJ    sille           cc              
sille           PRON     Varasin         obl             sekä
sisarentyttärelleni NOUN     Varasin         obl             pääsi, kaupunginteatterin
,               PUNCT    pääsi           punct           
joka            PRON     pääsi           nsubj           
pääsi           VERB     sisarentyttärelleni acl:relcl       ,, joka, Turkuun, lukemaan
Turkuun         PROPN    pääsi           obl             
lakia           NOUN     lukemaan        obj             
lukemaan        VERB     pääsi           advcl           lakia
,               PUNCT    liput           punct           
liput           NOUN     kaupunginteatterin nmod            ,
kaupunginteatterin NOUN     sisarentyttärelleni conj            liput
Laulavat        ADJ      sadepisarat     amod            
sadepisarat     NOUN     -musikaaliin    nsubj:cop       Laulavat
-musikaaliin    NOUN     ROOT            ROOT            sadepisarat, .
.               PUNCT    -musikaaliin    punct           

============================================================
Parser Evaluation - Finnish
============================================================

Evaluation Results (6506 tokens):
  Unlabeled Attachment Score (UAS): 74.56%
  Labeled Attachment Score (LAS):   62.40%
  Label Accuracy Score (LS):        77.90%

Visualization saved: dependency_analysis_Finnish.png

============================================================
Testing with Ambiguous Sentences
============================================================

Parsing ambiguous sentences...

============================================================
Example Dependency Parses - Finnish
============================================================

Example 1: Näen miehen kaukoputkella.
------------------------------------------------------------
Token           POS      Head            Dep Rel         Children
------------------------------------------------------------
Näen            VERB     ROOT            ROOT            kaukoputkella, .
miehen          NOUN     kaukoputkella   nmod:poss       
kaukoputkella   NOUN     Näen            obl             miehen
.               PUNCT    Näen            punct           

Example 2: Poliisi ampui miehen aseella.
------------------------------------------------------------
Token           POS      Head            Dep Rel         Children
------------------------------------------------------------
Poliisi         NOUN     ampui           nsubj           
ampui           VERB     ROOT            ROOT            Poliisi, aseella, .
miehen          NOUN     aseella         nmod:poss       
aseella         NOUN     ampui           obl             miehen
.               PUNCT    ampui           punct           

Example 3: Koira puraisee miestä puistossa.
------------------------------------------------------------
Token           POS      Head            Dep Rel         Children
------------------------------------------------------------
Koira           NOUN     puraisee        nsubj           
puraisee        VERB     ROOT            ROOT            Koira, miestä, puistossa, .
miestä          NOUN     puraisee        obj             
puistossa       NOUN     puraisee        obl             
.               PUNCT    puraisee        punct           

================================================================================
ANALYSIS COMPLETE!
================================================================================

Generated files:
  - dependency_analysis_Finnish.png

Note: Hindi parsing would require a trained model.
You can train your own using SpaCy or use alternative parsers.
(venv) shrinath@dellpro:~/ShriCode/Langugae-as-Data$ 