# IndicTTS Telugu Latin 'o' Typographical Anomaly Audit & Correction Report

**Total Utterances Audited:** 43  
**Total 'o' Glyphs Found:** 46  
**Distribution by Split:**
- Excluded Pool: 28 utterances
- Test Split: 13 utterances
- Validation Split: 1 utterances (`train_telugumale_01981`)
- Training Split: 1 utterances (`train_telugumale_02568`)

---

## 1. Executive Summary & Root Cause Analysis

During the Step 2.2A-R vocabulary audit, 43 utterances across the 8,576 IndicTTS Telugu corpus were identified containing the lowercase ASCII Latin character `'o'` (`U+006F`). All 43 utterances belong to the **male speaker** corpus (`train_telugumale_*`).

### Root Cause:
In Telugu orthography, the Anusvara (`ం`, `U+0C02`, known colloquially as *sunna* or zero) is represented graphically as a small circle. On standard transliteration keyboards (e.g., ITRANS, Baraha, Modular, Apple Indic, and phonetic keyboard layouts), transcribers frequently key the Latin lowercase letter `'o'` instead of the dedicated Telugu Anusvara key, or font encoding conversions erroneously decode the circular glyph as Latin `'o'`. Without correction, this character is treated as an out-of-vocabulary (`<unk>`) token in neural TTS and causes an out-of-vocabulary (OOV) lookup error in MFA forced alignment.

### Correction Policy:
1. **Never globally replace Latin 'o' without context**: Replacements must only target instances where `'o'` is immediately adjacent to Telugu Unicode characters (`[\u0C00-\u0C7F]`) or word boundaries.
2. **Preserve Source Data**: Original raw transcripts in manifests are retained; correction is applied deterministically during text normalization in `Telugu/text/normalizer.py`.
3. **Audit Logging**: Every single one of the 43 occurrences is cataloged below with before/after word comparisons and semantic verification.

---

## 2. Complete Utterance-by-Utterance Audit Table

| # | Utterance ID | Split | Original Word(s) | Corrected Word(s) | Semantic Meaning & Context |
| :---: | :--- | :---: | :--- | :--- | :--- |
| 1 | `train_telugumale_01475` | excluded | `వీళుoటే,` | `వీళుంటే,` | Anusvara typo in verb/noun suffix |
| 2 | `train_telugumale_01531` | test | `పేరేoటి` | `పేరేంటి` | Interrogative suffix -ఏంటి / -ఏంటంటే (what is) |
| 3 | `train_telugumale_01561` | excluded | `అనిపించిoది` | `అనిపించింది` | Anusvara typo in verb/noun suffix |
| 4 | `train_telugumale_01580` | test | `యిoది,` | `యింది,` | Anusvara typo in verb/noun suffix |
| 5 | `train_telugumale_01588` | excluded | `నoదరినీ` | `నందరినీ` | Anusvara typo in verb/noun suffix |
| 6 | `train_telugumale_01632` | test | `ఆశ్చర్యపోవటo` | `ఆశ్చర్యపోవటం` | Gerund suffix -డం/-టం (-ing) |
| 7 | `train_telugumale_01651` | excluded | `భోజనo,` | `భోజనం,` | భోజనం (meal / food) |
| 8 | `train_telugumale_01652` | excluded | `జ్ఞాపo` | `జ్ఞాపం` | Anusvara typo in verb/noun suffix |
| 9 | `train_telugumale_01665` | excluded | `వందలామoదిగా` | `వందలామందిగా` | Anusvara typo in verb/noun suffix |
| 10 | `train_telugumale_01667` | excluded | `వేడుకుoటె` | `వేడుకుంటె` | Anusvara typo in verb/noun suffix |
| 11 | `train_telugumale_01698` | excluded | `అoటు` | `అంటు` | Adverbial / participial form |
| 12 | `train_telugumale_01747` | test | `ఉoడేవాడు` | `ఉండేవాడు` | Auxiliary verb ఉండు (to be / to stay) |
| 13 | `train_telugumale_01752` | excluded | `చేశావoటే` | `చేశావంటే` | Anusvara typo in verb/noun suffix |
| 14 | `train_telugumale_01764` | test | `చేరుస్తుoడగా,` | `చేరుస్తుండగా,` | Anusvara typo in verb/noun suffix |
| 15 | `train_telugumale_01801` | excluded | `వెళ్ళిపోయిoది,` | `వెళ్ళిపోయింది,` | Anusvara typo in verb/noun suffix |
| 16 | `train_telugumale_01809` | excluded | `కొoత` | `కొంత` | Anusvara typo in verb/noun suffix |
| 17 | `train_telugumale_01842` | test | `దేవాళయo` | `దేవాళయం` | దేవాలయం (temple) |
| 18 | `train_telugumale_01858` | test | `దేవాళయo` | `దేవాళయం` | దేవాలయం (temple) |
| 19 | `train_telugumale_01861` | excluded | `రాకుమారులoదరికీ` | `రాకుమారులందరికీ` | Anusvara typo in verb/noun suffix |
| 20 | `train_telugumale_01862` | excluded | `కారణమేoటoటే` | `కారణమేంటంటే` | Interrogative suffix -ఏంటి / -ఏంటంటే (what is) |
| 21 | `train_telugumale_01878` | excluded | `ఇoకొకటి` | `ఇంకొకటి` | Anusvara typo in verb/noun suffix |
| 22 | `train_telugumale_01893` | excluded | `చేతికoదిస్తూ` | `చేతికందిస్తూ` | Anusvara typo in verb/noun suffix |
| 23 | `train_telugumale_01925` | test | `కారణమేoటి` | `కారణమేంటి` | Interrogative suffix -ఏంటి / -ఏంటంటే (what is) |
| 24 | `train_telugumale_01926` | excluded | `కారణమేoటి` | `కారణమేంటి` | Interrogative suffix -ఏంటి / -ఏంటంటే (what is) |
| 25 | `train_telugumale_01965` | excluded | `ఉoటారు,` | `ఉంటారు,` | Auxiliary verb ఉండు (to be / to stay) |
| 26 | `train_telugumale_01966` | test | `నడుస్తున్నదేoటి` | `నడుస్తున్నదేంటి` | Interrogative suffix -ఏంటి / -ఏంటంటే (what is) |
| 27 | `train_telugumale_01967` | excluded | `సంగతేoటంటే` | `సంగతేంటంటే` | Interrogative suffix -ఏంటి / -ఏంటంటే (what is) |
| 28 | `train_telugumale_01968` | test | `బ్రతికేవున్నాడoటున్నాడు` | `బ్రతికేవున్నాడంటున్నాడు` | Anusvara typo in verb/noun suffix |
| 29 | `train_telugumale_01981` | val | `అచ్చoగా` | `అచ్చంగా` | Adverbial / participial form |
| 30 | `train_telugumale_01999` | excluded | `దేవాళయo`, `దేవాళయoలోకి,` | `దేవాళయం`, `దేవాళయంలోకి,` | దేవాలయం (temple) |
| 31 | `train_telugumale_02016` | excluded | `రావడమేoటి,` | `రావడమేంటి,` | Interrogative suffix -ఏంటి / -ఏంటంటే (what is) |
| 32 | `train_telugumale_02039` | excluded | `ముహo` | `ముహం` | Anusvara typo in verb/noun suffix |
| 33 | `train_telugumale_02041` | excluded | `పట్టడo` | `పట్టడం` | Gerund suffix -డం/-టం (-ing) |
| 34 | `train_telugumale_02106` | excluded | `కీకారణ్యoల్ల్లో` | `కీకారణ్యంల్ల్లో` | Anusvara typo in verb/noun suffix |
| 35 | `train_telugumale_02204` | excluded | `ఉద్రేగoతో,` | `ఉద్రేగంతో,` | Anusvara typo in verb/noun suffix |
| 36 | `train_telugumale_02212` | test | `ఉoటే,` | `ఉంటే,` | Auxiliary verb ఉండు (to be / to stay) |
| 37 | `train_telugumale_02469` | test | `ఉoడేది,`, `ఉoడేది,` | `ఉండేది,`, `ఉండేది,` | Auxiliary verb ఉండు (to be / to stay) |
| 38 | `train_telugumale_02552` | excluded | `ఎoత` | `ఎంత` | Anusvara typo in verb/noun suffix |
| 39 | `train_telugumale_02563` | test | `వెళ్ళగొట్టడo,` | `వెళ్ళగొట్టడం,` | Gerund suffix -డం/-టం (-ing) |
| 40 | `train_telugumale_02568` | train | `చేస్తూవుoడేవాడు.` | `చేస్తూవుండేవాడు.` | Anusvara typo in verb/noun suffix |
| 41 | `train_telugumale_02711` | excluded | `ఊపయోగపడేoత` | `ఊపయోగపడేంత` | Anusvara typo in verb/noun suffix |
| 42 | `train_telugumale_02730` | excluded | `మార్గాంతo` | `మార్గాంతం` | Anusvara typo in verb/noun suffix |
| 43 | `train_telugumale_02814` | excluded | `ఆశ్రమo` | `ఆశ్రమం` | ఆశ్రమం (ashram / hermitage) |

---

## 3. Training & Validation Split Occurrences

Only two of the 43 occurrences fall into the active training and validation sets:

### 1. `train_telugumale_02568` (Training Split):
- **Original Text**:
  > `ఒకనాడు, అడవినుంచి బోధిసత్వుడు, ఇంటికి రాగానే, సునందుడు, అతని కొంకికర్ర, పళ్ళమూట అంది పుచ్చుకొని, ఇంటి లోపల చేరవేశాడు, అప్పటినుంచీ, వాడు శ్రద్దాభక్తులతో, బోధిసత్వుడి ఇంట, సమస్తమైన పనులు చేస్తూవుoడేవాడు.`
- **Target Word**: `చేస్తూవుoడేవాడు.` -> `చేస్తూవుండేవాడు.` (*used to do*)
- **Corrected Text**:
  > `ఒకనాడు, అడవినుంచి బోధిసత్వుడు, ఇంటికి రాగానే, సునందుడు, అతని కొంకికర్ర, పళ్ళమూట అంది పుచ్చుకొని, ఇంటి లోపల చేరవేశాడు, అప్పటినుంచీ, వాడు శ్రద్దాభక్తులతో, బోధిసత్వుడి ఇంట, సమస్తమైన పనులు చేస్తూవుండేవాడు.`
- **Verification**: Telugu verb form *చేస్తూ ఉండేవాడు* (he used to do) phonetically requires the anusvara preceding alveolar stop /ɖ/.

### 2. `train_telugumale_01981` (Validation Split):
- **Original Text**:
  > `సహదేవుడు అనుమతించి, తన వీణను తెచ్చి ఆమె ముందు పెట్టించాడు, వీణను శ్రుతిచేస్తూంటే వీణావతికి ఎక్కడలేని ఆనందము కలిగింది, అది అచ్చoగా తన వీణలాగే పలుకుతున్నది ఆమె ఆ అనందంలో అద్భుతంగా కచేరీ చేసింది.`
- **Target Word**: `అచ్చoగా` -> `అచ్చంగా` (*exactly / purely*)
- **Corrected Text**:
  > `సహదేవుడు అనుమతించి, తన వీణను తెచ్చి ఆమె ముందు పెట్టించాడు, వీణను శ్రుతిచేస్తూంటే వీణావతికి ఎక్కడలేని ఆనందము కలిగింది, అది అచ్చంగా తన వీణలాగే పలుకుతున్నది ఆమె ఆ అనందంలో అద్భుతంగా కచేరీ చేసింది.`
- **Verification**: Telugu adverb *అచ్చంగా* (purely, precisely) requires anusvara `ం` preceding velar /ɡaː/.

---

## 4. Normalizer Integration Rule

To resolve this issue cleanly across the pipeline without modifying raw source dataset files, the following contextual regular expression rule is added to `Telugu/text/normalizer.py`:

```python
# Replace Latin 'o' when used as a typographical error for Telugu Anusvara (ం)
_LATIN_O_ANUSVARA_REGEX = re.compile(r'(?<=[\u0C00-\u0C7F])o(?=[\u0C00-\u0C7F]|\b|[,\.!?;:"\'])')
text = _LATIN_O_ANUSVARA_REGEX.sub('ం', text)
```

This targeted contextual replacement guarantees that only Latin `'o'` characters embedded within Telugu words are converted to `ం`, preserving non-Telugu text intact.