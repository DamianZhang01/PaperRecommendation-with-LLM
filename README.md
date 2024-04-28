# Paper Recommendation with LLM

With the goal of democratizing access to academic research by simplifying the discovery

and understanding of scholarly articles across various domains, we leveraged the use of

three advanced algorithms to accurately extract corresponding word embeddings for all ab-

stracts of interest. The three algorithms include a transformer model using GPT2, a represen-

tation learning on large graphs using GraphSAGE, and a dual-metric recom-

mendation system developed by Microsoft. The proof of concept was applied

on a database of more than 600k academic papers. Consequently, we developed an accompa-

nying web-based application using Streamlit with the corresponding articles and embeddings

stored in SQLite and VecDB for faster retrieval and search similarity calculations. These imple-

mentations offer new ways of applying and integrating state-of-the-art large language models

in paper recommendation systems.

## Data

You can find the  transformed dataset [here](https://www.kaggle.com/datasets/zzydipper/citation-v1)

## Quick start

```bash
cd Application
streamlit run demo.py
```

