Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?
TensorFlow was developed by Google and emphasizes production, deployment, scalability, and compatibility with tools like TensorFlow Lite.
Uses static computation graphs.

PyTorch, developed by Meta, uses dynamic computation graphs, making it more flexible and intuitive for research & experimentation.

PyTorch → research, experimentation, and flexibility
TensorFlow → large-scale deployment

---

Q2: Describe two use cases for Jupyter Notebooks in AI development.
a) Model prototyping and experimentation – They allow data scientists to write and test small pieces of code interactively, visualize results immediately, and iterate quickly on model design.

b) Teaching, documentation, and sharing research –
Jupyter notebooks support Markdown, code, and visual outputs in one file, making them ideal for tutorials, project documentation, and collaborative research where results and explanations are combined.

---

Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?

It provides linguistically informed, high-performance NLP tools that go far beyond Python’s basic string handling.

Unlike plain string methods (like .split()), spaCy performs tokenization, part-of-speech tagging, named entity recognition, dependency parsing, and lemmatization using pretrained language models.

Provides deeper understanding of text structure and meaning, which is essential for tasks like sentiment analysis, text classification, and information extraction.

---

Comparative Analysis: Scikit-learn vs TensorFlow

Scikit-learn and TensorFlow differ mainly in their target applications, ease of use, and community support. Scikit-learn is primarily designed for classical machine learning algorithms such as linear and logistic regression, decision trees, clustering, and support vector machines. It is ideal for smaller datasets and quick experimentation with traditional models. In contrast, TensorFlow focuses on deep learning and neural networks, offering extensive support for large-scale data processing and hardware acceleration using GPUs and TPUs. It is more suitable for complex tasks like image recognition, natural language processing, and large-scale model deployment.

In terms of ease of use, Scikit-learn is generally simpler for beginners. It provides a clean, consistent API and requires minimal code to train and evaluate models. TensorFlow, while more powerful, has a steeper learning curve due to its emphasis on neural network construction, data pipelines, and computational graphs. However, TensorFlow 2.x has introduced a more user-friendly interface with Keras, making model development more intuitive.

Regarding community support, both libraries have large user bases, but their focus differs. Scikit-learn enjoys strong support in academic and industrial research communities dealing with classical machine learning problems. TensorFlow, backed by Google, benefits from an enormous ecosystem, comprehensive documentation, frequent updates, and wide adoption across enterprise-level AI and deep learning projects.
