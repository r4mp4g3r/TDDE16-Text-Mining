This project was done by Pachigulla Ramtej for the course TDDE16 Text Mining at Linköping University.

Abstract:
This paper presents a comprehensive multi-modal approach to fake news detection that
integrates RoBERTa text embeddings, CLIP image-text consistency analysis, and linguistic
feature extraction. The model was trained on the Fakeddit dataset, a large-scale collection of multi-modal news from Reddit. Evaluated on 6,892 test samples, the model achieved an
F1-score of 86.85% and 85.61% accuracy. An ablation study demonstrates that each component contributes meaningfully to the final performance, with the full model outperforming
a text-only baseline by 8.85 percentage points. An analysis of the model’s attention patterns
reveals that it learns to identify emotional manipulation and sensationalism—patterns that
align with human intuition about misinformation. These results validate the effectiveness of
multi-modal fusion for misinformation detection.