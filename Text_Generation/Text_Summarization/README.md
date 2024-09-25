# Text Summarization with SLMs (small language models)
* References, experiments and ideas using SLMs for text to text summarization.
* Cool links to checkout:
  1. [Text Summarization - BART vs. Pegasus vs. Longformer vs. BigBird vs. GPT](https://krishpro.github.io/text-summarization/)
  2. [Text Summarization - T5, BART, XLNET, GPT](https://towardsdatascience.com/summarize-reddit-comments-using-t5-bart-gpt-2-xlnet-models-a3e78a5ab944)


## Extractive vs. Abstractive Summarization
1. Extractive Summarization
   * This method‘extracts’ significant information from enormous amounts of text and arranges it into clear and succinct summaries. The approach is simple in that it extracts texts based on factors such the text to be summarized, the most essential sentences (Top K), and the importance of each of these phrases to the overall subject.
   * Can be subject to bias based on parameters specified.
2. Abstractive Summarization
   * This is what most deep learning and transformer models do.
   * Creates readable sentences from complete text input. Text is rewritten by producing acceptable representations, which are then analyzed and summarized.
   * Abstractive summarizations can be subject to errors the same as extractive because the text is also based on the underlying model and its training data.
  
References for Extractive vs. Abstractive Summarization Techniques
* [Mastering Text Summarization with SUMY](https://www.geeksforgeeks.org/mastering-text-summarization-with-sumy-a-python-library-overview/)
* [6 Useful Text Summarizaton Algorithms](https://medium.com/@sarowar.saurav10/6-useful-text-summarization-algorithm-in-python-dfc8a9d33074)
* [bert-extractive-summarizer repo](https://pypi.org/project/bert-extractive-summarizer/)
* [BART Models for text summarization](https://www.digitalocean.com/community/tutorials/bart-model-for-text-summarization-part1)
