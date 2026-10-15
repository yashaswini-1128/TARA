# Viva / Technical Q&A cheat sheet

**1. Why Case Study 3 and not the AUTOSAR HLD assistant?**
The evaluation rewards a data-science pipeline (dataset/EDA, feature engineering, model development, evaluation). A RAG app has no dataset to explore and no model to train or validate. CAN intrusion detection gives all of that and still produces a TARA, so it stays inside the case-study document.

**2. What problem does the project solve?**
CAN has no sender authentication, so attackers on the bus can flood, fuzz or spoof. We detect those attacks from traffic and attach the evidence to an ISO/SAE 21434-style risk assessment.

**3. Why is the risk rating not done by the LLM?**
Risk values are compliance artifacts and must be reproducible and auditable. Impact x feasibility comes from fixed tables. The LLM only drafts prose from the structured record and can be switched off.

**4. Which features matter most and why?**
Timing features (gap vs the ID's expected period) and ID behaviour (whitelist, rolling ID entropy). DoS is ID 0x000 at ~0.3 ms gaps; Fuzzy has random IDs and high payload entropy; Spoof attacks use legitimate IDs, so only timing/payload-repeat features catch them.

**5. How did you avoid data leakage?**
(a) Time-ordered split, not random. (b) ID whitelist and expected periods fitted on training-normal frames only. (c) Features use only past frames. (d) Group CV by time chunk so a burst is never split across folds. (e) Model chosen on CV, test used once.

**6. Why did shuffled CV give ~0.99 while the test set scored lower?**
Neighbouring frames in the same burst are near-identical; shuffled folds put them on both sides. Group CV fixes the estimate. Always explain this: it shows you validate honestly.

**7. How did you handle class imbalance?**
Class-weighted models, macro-F1 and per-class recall. SMOTE rejected because synthetic frames would invent timing patterns that never occur on a real bus.

**8. Why Random Forest / boosting?**
Tabular data, mixed scales, non-linear interactions, fast CPU inference, built-in importances. Compared against LR, KNN, Decision Tree, SVM. Final choice by group-CV macro-F1.

**9. What does PCA do here, and why not use it in the final model?**
It keeps 95% variance with fewer components, but components lose meaning. A TARA justification needs named features, so we use feature selection for interpretability and show PCA as a comparison.

**10. What is the zero-day experiment?**
Leave one attack family out of training and measure how many of its frames are still flagged. Supervised models can miss unseen families; Isolation Forest (trained on normal only) needs no attack labels. Production: use both.

**11. Which metric matters most?**
Attack recall (a missed attack is costly) together with false-alarm rate (alert fatigue). Accuracy alone is misleading with 85%+ normal traffic.

**12. How is it deployed?**
Same `src/` code serves FastAPI (`/predict`, `/tara`) and Streamlit. Dockerfile bundles both. Artifacts are versioned; PSI drift monitor triggers review.

**13. What are the limitations?**
Synthetic data unless retrained on real captures, four attack families, no adversarial-evasion testing, illustrative TARA tables, not validated for production vehicles.

**14. How would you extend it?**
Real HCRL / CICIoV data, CAN-FD, sequence models (LSTM/autoencoder), joint HARA-TARA graph, online learning with analyst feedback, edge deployment on an automotive gateway.

**15. Ethics / responsible AI?**
Human approval, no autonomous action on vehicles, labelled illustrative standards content, privacy of architecture data, documented limitations in the model card.

**16. Did you iterate on the features? What did error analysis show?**
Yes. Spoof_Gear was missed in several runs: every forged burst uses a different frozen payload, so payload values cannot generalise. I added `payload_ham_min` (fewest differing bytes vs the last 4 same-ID frames), which is value-agnostic. Over 8 unseen seeds macro F1 went from 0.841 to 0.972, and to 0.987 with the payload-change features. Two payload-change features alone gave no gain, so I tested combinations instead of assuming. `scripts/feature_ablation.py` reproduces it.

**17. Why trust a result from one simulator?**
I don't fully: simulator seeds vary the burst timing and payloads, but the attack models are my own. That is why the loader for the real HCRL files exists and why synthetic scores are labelled as such.
