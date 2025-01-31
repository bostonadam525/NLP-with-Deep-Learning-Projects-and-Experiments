# Topic extraction from PDF experiment
* This uses a python library pymupdf4llm to transform an open source PDF document into markdown and then markdown to text.
* Chunking of the text is performed and embedded using an open source model from hugging face.
* Then Topic modeling using BERTopic with Llama-3.1-8B-instruct is performed.
* The document used is open source via this website: https://investors.modernatx.com/events-and-presentations/events/default.aspx
