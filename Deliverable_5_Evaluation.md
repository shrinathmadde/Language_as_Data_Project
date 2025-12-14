Using device: cuda
======================================================================
DELIVERABLE 5: HUMAN EVALUATION OF MODEL OUTPUT
======================================================================

================================================================================
ANNOTATION GUIDELINES FOR LANGUAGE MODEL EVALUATION
================================================================================

TASK: Evaluate generated text continuations from two language models (A and B).

--------------------------------------------------------------------------------
CATEGORY 1: SYNTACTIC FLUENCY (1-5 scale)
--------------------------------------------------------------------------------
Evaluate whether the generated text follows grammatical rules of the language.

Score | Description
------|---------------------------------------------------------------------------
  1   | Completely ungrammatical; random word sequences; incomprehensible
  2   | Mostly ungrammatical; major errors in word order, agreement, or case
  3   | Some grammatical errors but structure is recognizable
  4   | Minor grammatical errors; mostly well-formed sentences
  5   | Fully grammatical; native-like sentence structure

Examples (Hindi):
- Score 5: "प्रधानमंत्री ने कहा कि सरकार इस मुद्दे पर विचार करेगी।"
- Score 3: "प्रधानमंत्री कहा है कि सरकार मुद्दे विचार।"
- Score 1: "प्रधानमंत्री है कहा ने सरकार इस।"

Examples (Finnish):
- Score 5: "Hän on ollut professorina yliopistossa viisi vuotta."
- Score 3: "Hän on ollut professori yliopisto viisi vuotta."
- Score 1: "Hän professori ollut on yliopisto."

--------------------------------------------------------------------------------
CATEGORY 2: SEMANTIC COHERENCE (1-5 scale)
--------------------------------------------------------------------------------
Evaluate whether the text makes sense and maintains a logical topic/theme.

Score | Description
------|---------------------------------------------------------------------------
  1   | Completely incoherent; no logical connection between words
  2   | Mostly incoherent; topic shifts randomly; contradictory statements
  3   | Partially coherent; some logical flow but with confusing parts
  4   | Mostly coherent; clear topic with minor inconsistencies
  5   | Fully coherent; logical flow; meaningful content that follows from prompt

Examples (Hindi):
- Score 5: "यह एक ऐतिहासिक इमारत है जो मुगल काल में बनाई गई थी।"
- Score 3: "यह एक ऐतिहासिक इमारत है जो खाना बनाती है।"
- Score 1: "यह एक ऐतिहासिक नीला चलना पानी कल।"

Examples (Finnish):
- Score 5: "Tämä on vanha rakennus, joka rakennettiin 1800-luvulla."
- Score 3: "Tämä on vanha rakennus, joka syö kalaa."
- Score 1: "Tämä on vanha sininen juosta vesi huomenna."

--------------------------------------------------------------------------------
CATEGORY 3: OVERALL PREFERENCE (A/B/Tie)
--------------------------------------------------------------------------------
Which output would you prefer to read? Consider both fluency and coherence.

- A: Output A is clearly better
- B: Output B is clearly better
- Tie: Both outputs are roughly equal in quality

--------------------------------------------------------------------------------
HANDLING <UNK> TOKENS
--------------------------------------------------------------------------------
- <UNK> (unknown word) tokens indicate words not in vocabulary
- Many <UNK> tokens should lower both fluency AND coherence scores
- A sentence with 50%+ <UNK> tokens: maximum score of 2 for both categories

--------------------------------------------------------------------------------
ANNOTATION PROCESS
--------------------------------------------------------------------------------
1. Read the prompt to understand the expected continuation context
2. Read Output A - score fluency (1-5), then coherence (1-5)
3. Read Output B - score fluency (1-5), then coherence (1-5)
4. Decide overall preference (A/B/Tie)
5. Add notes for unusual cases or observations

--------------------------------------------------------------------------------
FOR NON-NATIVE SPEAKERS
--------------------------------------------------------------------------------
If you don't speak the target language:
1. Use Google Translate to get rough English translations
2. Check grammar using online tools (e.g., LanguageTool)
3. Focus on structural patterns you can identify
4. Mark uncertain ratings with "?" in notes column
5. Document your translation/verification process

================================================================================

Guidelines saved: annotation_guidelines.txt

======================================================================
EVALUATION PIPELINE: HINDI
======================================================================

Loading data...
Vocabulary size: 9124

Training Model A (Baseline)...
  Epoch 1: Train Loss=4.5608, Dev Loss=4.3526
  Epoch 2: Train Loss=3.6389, Dev Loss=4.4273
  Epoch 3: Train Loss=3.2831, Dev Loss=4.5815
  Epoch 4: Train Loss=3.0706, Dev Loss=4.7274
  Epoch 5: Train Loss=2.9230, Dev Loss=4.8446
  Epoch 6: Train Loss=2.8110, Dev Loss=4.9630

Training Model B (Label Smoothing)...
  Epoch 1: Train Loss=5.2184, Dev Loss=4.9948
  Epoch 2: Train Loss=4.5270, Dev Loss=4.9685
  Epoch 3: Train Loss=4.2672, Dev Loss=4.9966
  Epoch 4: Train Loss=4.1059, Dev Loss=5.0286
  Epoch 5: Train Loss=3.9913, Dev Loss=5.0589
  Epoch 6: Train Loss=3.9021, Dev Loss=5.0888
  Epoch 7: Train Loss=3.8307, Dev Loss=5.1124

Generating evaluation samples...
Samples saved: evaluation_samples_Hindi.json
Annotation sheet saved: annotation_sheet_Hindi.csv

======================================================================
GENERATED SAMPLES FOR HINDI
======================================================================

[1] Prompt: प्रधानमंत्री ने
    Model A (Baseline):        प्रधानमंत्री ने सीईसीए को आज अस्पताल के निकट एक व्यक्ति की गोली मारकर हत्या कर दी ।
    Model B (Label Smoothing): प्रधानमंत्री ने उनके खिलाफ परमाणु हथियार फ्रंट ( जेकेएलएफ ) में अन्य को छोड़ दी है ।

[2] Prompt: सरकार ने
    Model A (Baseline):        सरकार ने जम्मू - कश्मीर के पहलगाम के इलाकों में भी गए हैं ।
    Model B (Label Smoothing): सरकार ने कहा था कि हम देखना चाहते हैं , तो वे कह सकते हैं कि क्या उनके कांग्रेसी नेता हैं ।

[3] Prompt: पुलिस ने
    Model A (Baseline):        पुलिस ने उसे गोली मार दी थी ।
    Model B (Label Smoothing): पुलिस ने जनमेजय को स्वस्थ करने का आदेश दिया ।

[4] Prompt: अधिकारियों ने
    Model A (Baseline):        अधिकारियों ने बताया कि इन दोनों भाई <UNK> <UNK> ने ही कर लिया ।
    Model B (Label Smoothing): अधिकारियों ने बताया कि इस मामले में न्यायमूर्ति बी. आई. १९७१ में शामिल हुए ।

[5] Prompt: यह एक
    Model A (Baseline):        यह एक अनूठा संग्रहालय है जो 200 एकड़ का आकार कैसे है ।
    Model B (Label Smoothing): यह एक संवेदनशील मुद्दा है ।

[6] Prompt: वह बहुत
    Model A (Baseline):        वह बहुत <UNK> की वजह से दुख जाता है ।
    Model B (Label Smoothing): वह बहुत से <UNK> बातें मौकों पर थे ।

[7] Prompt: यहां पर
    Model A (Baseline):        यहां पर आ गई पाबंदी और सट्टेबाजी में छापे मारे गए थे ।
    Model B (Label Smoothing): यहां पर <UNK> नगर दास द्वारा चलाई जा रही संस्थान के वकील डी. <UNK> ने पुरस्कार ग्रहण करने पर बात की है

[8] Prompt: इस समय
    Model A (Baseline):        इस समय पूर्व प्रधानमंत्री मनमोहन सिंह और पाकिस्तान ने भी वरिष्ठ नेता पद से इस्तीफा दे दिया ।
    Model B (Label Smoothing): इस समय मंगलवार देर रात केवल भारत <UNK> में लगा देते हैं ।

[9] Prompt: उन्होंने कहा
    Model A (Baseline):        उन्होंने कहा कि कश्मीर के किसान पर विचार कर रही है कि वे अपने विदेशी राज्य में एक समान फीस हो सकती
    Model B (Label Smoothing): उन्होंने कहा कि यह भी चमत्कार से कम नहीं है , लेकिन सुप्रीम कोर्ट का कहना है कि देश की परियोजनाओं का

[10] Prompt: लोगों ने
    Model A (Baseline):        लोगों ने इसके खिलाफ चलने को कम की <UNK> देने का सुझाव दिया ।
    Model B (Label Smoothing): लोगों ने २००६ से ही क्यों जारी कर दी ।

[11] Prompt: हमें इस
    Model A (Baseline):        हमें इस तरह की राय न करने के लिए राष्ट्रपति मनमोहन सिंह को बुलाकर <UNK> रहे थे कि वह भारत से कुछ
    Model B (Label Smoothing): हमें इस तरह की बेटी और उनके परिवार और समूह के बीच सहयोग के लिए एक अन्य देश के तमाम जिलों को

[12] Prompt: आज के
    Model A (Baseline):        आज के कार्यकाल में भारत - <UNK> के विमानों के डीएम से राज्य में भारत के लोग काफी संख्या में महत्वपूर्ण हैं
    Model B (Label Smoothing): आज के गृह में अन्य पर <UNK> को पूरा करना पड़ा ।

[13] Prompt: क्या आप
    Model A (Baseline):        क्या आप किसी भी नहीं है ।
    Model B (Label Smoothing): क्या आप हुए हैं और इस तरह की कोई पहल नहीं की जा सकती ।

[14] Prompt: जब वह
    Model A (Baseline):        जब वह अपना साथ देने का कोई फैसला नहीं होगा ।
    Model B (Label Smoothing): जब वह तय हो गया जब वे अपनी बर्खास्तगी को ध्यान में रखकर नहीं मिला ।

[15] Prompt: अगर हम
    Model A (Baseline):        अगर हम हालात के समाधान के लिए सुरक्षा तैयारियां व <UNK> , <UNK> , जेड , <UNK> और <UNK> <UNK> को <UNK>
    Model B (Label Smoothing): अगर हम इस पर चिंतित नजर आते हैं ।

[16] Prompt: इसके बाद
    Model A (Baseline):        इसके बाद ही यह बेहद <UNK> है ।
    Model B (Label Smoothing): इसके बाद सरकार ने घरेलू , १९७६ <UNK> की <UNK> , <UNK> , <UNK> , <UNK> , मुकेश , <UNK> आदि सुरेश

======================================================================
EVALUATION PIPELINE: FINNISH
======================================================================

Loading data...
Vocabulary size: 14091

Training Model A (Baseline)...
  Epoch 1: Train Loss=4.9639, Dev Loss=4.4165
  Epoch 2: Train Loss=3.8754, Dev Loss=4.6339
  Epoch 3: Train Loss=3.3060, Dev Loss=4.9303
  Epoch 4: Train Loss=2.9565, Dev Loss=5.2169
  Epoch 5: Train Loss=2.7165, Dev Loss=5.4752
  Epoch 6: Train Loss=2.5381, Dev Loss=5.6851

Training Model B (Label Smoothing)...
  Epoch 1: Train Loss=5.6487, Dev Loss=5.0992
  Epoch 2: Train Loss=4.8223, Dev Loss=5.1892
  Epoch 3: Train Loss=4.3976, Dev Loss=5.2854
  Epoch 4: Train Loss=4.1282, Dev Loss=5.3725
  Epoch 5: Train Loss=3.9373, Dev Loss=5.4347
  Epoch 6: Train Loss=3.7915, Dev Loss=5.4994

Generating evaluation samples...
Samples saved: evaluation_samples_Finnish.json
Annotation sheet saved: annotation_sheet_Finnish.csv

======================================================================
GENERATED SAMPLES FOR FINNISH
======================================================================

[1] Prompt: hän on
    Model A (Baseline):        hän on varma että <UNK> <UNK> niitä , kuten pystyn <UNK> <UNK> .
    Model B (Label Smoothing): hän on ollut professorina työ , on kyllä paljon tulossa kuin eurooppalaisen <UNK> .

[2] Prompt: se oli
    Model A (Baseline):        se oli kamalaa , joka laajeni <UNK> mccainin <UNK> .
    Model B (Label Smoothing): se oli 1800-luvun aikana hyvin .

[3] Prompt: minä olen
    Model A (Baseline):        minä olen toisenlainen .
    Model B (Label Smoothing): minä olen varma , että en ole , mutta se on joku <UNK> .

[4] Prompt: he ovat
    Model A (Baseline):        he ovat tyytyväisiä <UNK> .
    Model B (Label Smoothing): he ovat tyytyväisiä toistensa <UNK> kanssa .

[5] Prompt: suomen hallitus
    Model A (Baseline):        suomen hallitus on keskittynyt <UNK> seurojen <UNK> , <UNK> , <UNK> ja <UNK> ja stora <UNK> <UNK> .
    Model B (Label Smoothing): suomen hallitus on <UNK> ja päivitetty .

[6] Prompt: presidentti sanoi
    Model A (Baseline):        presidentti sanoi viime vuonna <UNK> <UNK> jälkeen tammikuusta 2007 oli <UNK> prosenttia .
    Model B (Label Smoothing): presidentti sanoi nykyisen <UNK> ja <UNK> <UNK> <UNK> <UNK> turun yliopiston <UNK> tutkimuskeskuksesta toteavat .

[7] Prompt: poliisi on
    Model A (Baseline):        poliisi on käyttänyt <UNK> <UNK> eurooppaa <UNK> .
    Model B (Label Smoothing): poliisi on siirtynyt sosiaalisesta <UNK> <UNK> , hp:n <UNK> konsuli kesken , <UNK> ja <UNK> , joka ei <UNK> tai <UNK> ,

[8] Prompt: tutkijat ovat
    Model A (Baseline):        tutkijat ovat <UNK> yhteistyöhön , kaikki <UNK> <UNK> <UNK> .
    Model B (Label Smoothing): tutkijat ovat nopeasti <UNK> , joten <UNK> ehdokkaiden <UNK> ja <UNK> <UNK> <UNK> <UNK> .

[9] Prompt: tämä on
    Model A (Baseline):        tämä on ryhtynyt hyvin .
    Model B (Label Smoothing): tämä on tehnyt limittäin <UNK> <UNK> <UNK> .

[10] Prompt: siellä oli
    Model A (Baseline):        siellä oli vaikuttava vasta <UNK> , kun se syttyi suunnilleen <UNK> , kertoi latvala päivän lopulla .
    Model B (Label Smoothing): siellä oli aika <UNK> <UNK> .

[11] Prompt: nyt on
    Model A (Baseline):        nyt on hyvä hetki , että ei sitä ole ollut <UNK> ?
    Model B (Label Smoothing): nyt on <UNK> mukaan .

[12] Prompt: ensi vuonna
    Model A (Baseline):        ensi vuonna <UNK> <UNK> <UNK> helpottaa tiedonkeruuta ja <UNK> .
    Model B (Label Smoothing): ensi vuonna päätimme olla <UNK> valmistuksessa .

[13] Prompt: mutta hän
    Model A (Baseline):        mutta hän oli jättänyt <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> eli <UNK> kanssa .
    Model B (Label Smoothing): mutta hän käveli minut töihin .

[14] Prompt: koska se
    Model A (Baseline):        koska se ei välttämättä pääse <UNK> <UNK> , ja siihen sovelletaan sellaisenaan kaikissa jäsenvaltioissa .
    Model B (Label Smoothing): koska se ei <UNK> .

[15] Prompt: kun hän
    Model A (Baseline):        kun hän on <UNK> käsivartensa <UNK> tai <UNK> .
    Model B (Label Smoothing): kun hän itse minulle <UNK> , uni ja <UNK> .

[16] Prompt: jos me
    Model A (Baseline):        jos me olemme , on ihan tähän sanoa , oikeastaan se on tullut , että sitä ei ole hyvin <UNK> .
    Model B (Label Smoothing): jos me että minua on eri ajassa ku tää <UNK> .

