
1. Label presence (Figure 1)
2. Label frequency and scope (Figure 2)
3. Labels per sentence (Table 1)
4. Lexical overlap (Figure 3)
5. Co-occurrence (Figure 4)
6. Distinctive vocabulary (Table 2)
7. Token lengths (Figures 5 and 6)
### Most sentences describe no technique

The distribution shows that 79% of the sentences in the dataset are unlabeled. This is consistent across splits where the in-scope labeled sentences are about 20.8, 22.4, and 20.0% for train, dev, and test, respectively. 

![Labeled vs unlabeled sentences per split](eda_results/label_presence.png)
Figure 1: Labeled vs unlabeled sentence distributions for each dataset

Because the retriever ranks techniques for every sentence, predicting its top candidates for all sentences would make most predictions false positives, capping precision near 21% even with a perfect ranking. Accuracy is uninformative, since predicting nothing would be correct 79% of the time, so we evaluate with precision, recall, and F1-score. Both stages use thresholds that allow a sentence to receive no prediction. The verifier is trained on negative candidates that include top-ranked techniques for unlabeled sentences, so it learns to reject them. We expect it to raise precision by more than it lowers recall by learning to reject incorrect sentence–technique candidate pairings. These pairs that are rejected are hopefully the false positives that the retrieval model has selected leading to increased precision and f1-score. 

### Label Frequency and Scope

Technique frequency is highly uneven. The five most frequent techniques, led by T1027 (Obfuscated Files or Information, 475 training sentences) and T1140 (Deobfuscate/Decode Files or Information, 347), account for 39.7% of all training labels, while the median in-scope technique has 54.5 training sentences.

We restrict evaluation to the 40 techniques with at least 20 training sentences, since techniques with fewer examples cannot be learned or evaluated reliably. The 10 excluded techniques have between 5 and 16 training sentences each. The 40 in-scope techniques account for 95.2% of test labels, so the restriction removes little of the evaluation data.

![Training sentences per technique](eda_results/technique_frequency.png)  
Figure 2: Training sentences per technique. Blue bars are in scope; gray bars fall below the 20-sentence cutoff (dashed line).

Because a few techniques dominate the labels, micro-averaged F1, which pools all predictions, would mostly reflect performance on those frequent techniques. We therefore also report macro-averaged F1, which averages the F1 score of each technique equally.

### Labels per Sentence

Most labeled sentences describe a single technique, but a substantial share describe more than one.

|Split|No label|1 label|2 labels|3+ labels|Labeled sentences with 2+ labels|
|---|---|---|---|---|---|
|Train|10,565|2,327|448|96|18.9%|
|Dev|1,929|444|96|34|22.6%|
|Test|2,368|464|125|32|25.3%|

Table 1: Number of technique labels per sentence.

Because nearly one in five labeled training sentences carries more than one technique, the retriever’s loss must allow several correct techniques per sentence. A standard contrastive loss assumes one correct match, and would push a sentence’s own labeled techniques apart from each other. Our loss compares each labeled technique only against techniques the sentence is not labeled with.

### Lexical Overlap Between Sentences and Technique Descriptions

To test whether techniques can be identified by shared words alone, we measured word overlap between each labeled sentence and the technique descriptions. Both texts were lowercased, lemmatized, and stripped of stopwords and punctuation. Overlap is the Jaccard similarity, the number of words the two texts share divided by the number of distinct words across both:

$$J(s, d) = \frac{|W_s \cap W_d|}{|W_s \cup W_d|}$$

where $W_s$ and $W_d$ are the word sets of sentence $s$ and description $d$.

For each labeled sentence, we computed its Jaccard similarity with all 40 in-scope technique descriptions and ranked the techniques from highest to lowest score. A sentence with two labels contributes two sentence–label pairs, giving 3,435 pairs in total.

|Measure|Value|
|---|---|
|Median Jaccard similarity with the labeled technique|0.036|
|Pairs sharing no words with the labeled technique’s description|18.5%|
|Labeled technique has the highest score of the 40|19.1%|
|Labeled technique among the five highest scores|48.0%|
|Median rank of the labeled technique|6|

![Word overlap between sentences and technique descriptions](eda_results/lexical_overlap.png)  
Figure 3: Left, Jaccard similarity between each sentence and its labeled technique’s description. Right, the rank of the labeled technique among the 40 in-scope techniques when ranked by Jaccard similarity.

Selecting the technique with the most word overlap would choose the labeled technique in fewer than one in five cases. The gap varies by technique. For T1105 (Ingress Tool Transfer), the labeled technique has the highest score in only 4.0% of 176 pairs, and for T1547 (Boot or Logon Autostart Execution) in 6.0% of 50, consistent with TTPrint’s observation that reports describe Run-key persistence in words that barely overlap with T1547’s definition. At the other end, T1204 (User Execution) reaches 56.8%. Reports and ATT&CK descriptions mostly express the same behavior in different words, which is why the retriever is trained to match sentences to descriptions rather than relying on shared vocabulary.


## Technique Co-occurrence 

The most frequent pair is T1027 (Obfuscated Files or Information) and T1140 (Deobfuscate/Decode Files or Information), labeled together on 72 training sentences. A single sentence often describes both halves of one sequence, an obfuscated payload that the malware decodes at runtime, so the two are likely to be confused and are a focus for error analysis. 

![Technique co-occurrence in training sentences](eda_results/cooccurrence_heatmap.png)
Figure 4: Correlation matrix of Technique's by sentence level co-occurrence (train dataset)
## Distinctive Vocabulary Exists

The most distinctive words by each technique are extracted from the data via TF-IDF. The table below show the words that correlate most by technique. 

| technique | name                                   | top raw                                                                                      | top by frequency                                                                                                                               | top by TFIDF <br>(count, weight, df=N)                                                                                                                                                                                                                              |
| --------- | -------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| T1003     | OS Credential Dumping                  | lsass, credentials, mimikatz, process, credential, technique, dump, memory, observed, actors | lsass (50), credentials (38), mimikatz (34), process (31), credential (27), technique (23), dump (21), memory (18), observed (16), actors (16) | mimikatz (34, 0.13, df=9), subsystem (7, 0.12, df=2), lsass (50, 0.12, df=12), lsass.exe (5, 0.12, df=1), authority (7, 0.11, df=3), comsvcs.dll (5, 0.11, df=2), dump (21, 0.11, df=11), dumping (10, 0.10, df=7), stores (6, 0.10, df=4), harvest (4, 0.10, df=2) |
| T1005     | Data from Local System                 | data, information, files, system, technique, stealer, victim, local, sensitive, file         | data (27), information (23), files (20), system (13), technique (13), stealer (11), victim (11), local (11), sensitive (10), file (9)          | nemesis (3, 0.10, df=2), stealer (11, 0.09, df=14), sensitive (10, 0.08, df=14), record (3, 0.08, df=4), steal (5, 0.08, df=9), fig (3, 0.08, df=5), 1033 (2, 0.08, df=2), english (2, 0.08, df=2), collects (7, 0.08, df=13), drives (4, 0.08, df=8)               |
| T1016     | System Network Configuration Discovery | ip, technique, information, system, network, address, winnti, account, source, host          | ip (22), technique (22), information (17), system (16), network (15), address (12), winnti (10), account (10), source (10), host (9)           | systeminfo (7, 0.13, df=5), ipconfig (7, 0.13, df=5), /all (4, 0.11, df=4), plan (5, 0.11, df=6), source (10, 0.11, df=12), triage (3, 0.10, df=3), develop (5, 0.10, df=7), returns (7, 0.10, df=10), platform (4, 0.10, df=6), none (4, 0.10, df=6)               |
| T1021     | Remote Services                        | rdp, remote, desktop, actors, network, using, used, technique, threat, connections           | rdp (44), remote (34), desktop (26), actors (19), network (18), using (17), used (17), technique (17), threat (15), connections (15)           | et (5, 0.12, df=1), rdp (44, 0.10, df=18), netsh (4, 0.09, df=3), move (7, 0.09, df=8), desktop (26, 0.08, df=20), sessions (6, 0.08, df=8), dfir (2, 0.08, df=1), enable=yes (2, 0.08, df=1), rdpclip.exe (2, 0.08, df=1), establishing (4, 0.08, df=6)            |
Table 2: TF-IDF analysis and feature extraction for each technique

The distinctive-word lists mix genuine technique vocabulary (`mimikatz`, `hollowing`, `decrypt`) with report-specific artifacts, which is another illustration of why the models are evaluated on report-level splits. A model that learned `//factura61` as evidence for T1027 would look accurate on sentences from that same report and fail everywhere else.

## Token Lengths

Both encoders (MiniLM and SecureBERT) accept at most 512 tokens, and longer inputs raise an error, so all inputs are truncated. Several sentences exceed this limit, the longest reaching 1,126 tokens under MiniLM. Within that limit, we truncate sentences at 128 tokens, which keeps 97.3% of sentences intact under MiniLM and 97.8% under SecureBERT while reducing training time and memory relative to longer inputs. The truncated sentences are mostly long non-prose blocks such as indicator lists, so truncating them loses little information. Technique texts reach at most 155 tokens and are never truncated at the 256-token cutoff.

![Token lengths of technique texts](eda_results/token_lengths_techniques.png)
Figure 5: Technique Cutoffs by token length

### Verifier Input Length

The verifier reads each sentence and technique description together as a single input, so their combined length must fit within the encoder’s 512-token limit. Across all training sentence–label pairs, the combined length has a median of [median] tokens, and [pct_within_cutoff]% of pairs fit within 512; when a pair exceeds the limit, the sentence is truncated and the description is kept whole.

![Token lengths of sentences](eda_results/token_sentences.png)
Figure 6: Counts of sentence lengths by token

The table shows the number of sentence truncated. There are some outliers as the distribution of sentence lengths is largely right tailed and makes up less than 3 percent of the overall distribution. 

| input          | tokenizer  | median | p95 | p99 |  max | cutoff | pct_within_cutoff | n_truncated |
| -------------- | ---------- | -----: | --: | --: | ---: | -----: | ----------------: | ----------: |
| sentence       | MiniLM     |     28 |  86 | 249 | 1126 |    128 |             97.31 |         362 |
| sentence       | SecureBERT |     26 |  79 | 223 |  998 |    128 |             97.75 |         302 |
| technique text | MiniLM     |     85 | 137 | 152 |  155 |    256 |               100 |           0 |
| technique text | SecureBERT |     75 | 125 | 136 |  142 |    256 |               100 |           0 |
Table 4: Breakdown of cutoffs for tokens and percent truncated


---

### Summary

The exploratory analysis establishes five properties of the data, each of which shapes the system design:

|Finding|Design consequence|
|---|---|
|79% of sentences describe no technique, consistently across splits|Both stages use thresholds that allow a sentence to receive no prediction; evaluation uses precision, recall, and F1 rather than accuracy; the verifier is trained to reject candidates for unlabeled sentences|
|Technique frequency is highly uneven; 40 of 50 techniques have at least 20 training sentences and cover 95.2% of test labels|Evaluation is restricted to those 40 techniques, and macro-F1 is reported alongside micro-F1|
|18.9% of labeled training sentences carry more than one technique|The retriever’s loss allows multiple correct techniques per sentence|
|Sentences share few words with their labeled technique’s description, which has the highest word overlap in only 19.1% of cases|Techniques cannot be identified by keyword matching; the retriever is trained to match sentences to descriptions|
|Frequently co-labeled techniques, such as T1027 and T1140, and report-specific vocabulary appear in the data|Co-labeled pairs are a focus for error analysis; report-level splits prevent models from scoring well by memorizing report-specific words|
Table 5: SUmmary of Exploratory analysis findings. 

The input lengths support the chosen configuration. A 128-token sentence cutoff keeps over 97% of sentences intact under both tokenizers, and no technique description is truncated. The analysis also found technique and tactic IDs written directly in some sentences, which would let a model read the answer from the input; these are masked during preprocessing.