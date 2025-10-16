## Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?

TensorFlow was developed by Google and emphasizes production, deployment, scalability, and compatibility with tools like TensorFlow Lite.
Uses static computation graphs.

PyTorch, developed by Meta, uses dynamic computation graphs, making it more flexible and intuitive for research & experimentation.

PyTorch → research, experimentation, and flexibility
TensorFlow → large-scale deployment

---

## Q2: Describe two use cases for Jupyter Notebooks in AI development.

a) Model prototyping and experimentation – They allow data scientists to write and test small pieces of code interactively, visualize results immediately, and iterate quickly on model design.

b) Teaching, documentation, and sharing research –
Jupyter notebooks support Markdown, code, and visual outputs in one file, making them ideal for tutorials, project documentation, and collaborative research where results and explanations are combined.

---

## Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?

It provides linguistically informed, high-performance NLP tools that go far beyond Python’s basic string handling.

Unlike plain string methods (like .split()), spaCy performs tokenization, part-of-speech tagging, named entity recognition, dependency parsing, and lemmatization using pretrained language models.

Provides deeper understanding of text structure and meaning, which is essential for tasks like sentiment analysis, text classification, and information extraction.

---

## Comparative Analysis: Scikit-learn vs TensorFlow

Scikit-learn and TensorFlow differ mainly in their target applications, ease of use, and community support. Scikit-learn is primarily designed for classical machine learning algorithms such as linear and logistic regression, decision trees, clustering, and support vector machines. It is ideal for smaller datasets and quick experimentation with traditional models. In contrast, TensorFlow focuses on deep learning and neural networks, offering extensive support for large-scale data processing and hardware acceleration using GPUs and TPUs. It is more suitable for complex tasks like image recognition, natural language processing, and large-scale model deployment.

In terms of ease of use, Scikit-learn is generally simpler for beginners. It provides a clean, consistent API and requires minimal code to train and evaluate models. TensorFlow, while more powerful, has a steeper learning curve due to its emphasis on neural network construction, data pipelines, and computational graphs. However, TensorFlow 2.x has introduced a more user-friendly interface with Keras, making model development more intuitive.

Regarding community support, both libraries have large user bases, but their focus differs. Scikit-learn enjoys strong support in academic and industrial research communities dealing with classical machine learning problems. TensorFlow, backed by Google, benefits from an enormous ecosystem, comprehensive documentation, frequent updates, and wide adoption across enterprise-level AI and deep learning projects.

========================================================

## **\* Part 3 — Ethics & Optimization**

## 1. Ethical Considerations (bias):

MNIST bias: images are grayscale, well-centered, and mostly Western digits — models trained only on MNIST may perform poorly on digits written with different styles, scripts, or backgrounds. Mitigation: evaluate on extended datasets (e.g., EMNIST), apply data augmentation, and monitor subgroup performance. Tools like TensorFlow Fairness Indicators can slice metrics across cohorts (e.g., writer ID, stroke thickness) if metadata exists.

Amazon Reviews bias: language varies by region; reviews can reflect cultural or gendered language patterns. A simple rule-based sentiment can over-penalize objective mentions (“not worth for me”) or domain-specific slang. Mitigation: build transparent rules with error analysis, expand lexicons, or complement with calibrated ML models; audit performance across product categories/brands and manually sample edge cases.

## 2. Troubleshooting (common DL bugs):

Dimension mismatch: For CNNs, track shapes after each layer. Use prints or torchinfo.summary. Ensure in_features of the first Linear matches C*H*W after the final pooling.

Wrong loss: For multi-class classification with logits, use CrossEntropyLoss (PyTorch) or SparseCategoricalCrossentropy(from_logits=True) (TF). Do not apply softmax before those losses.

Wrong labels dtype: In PyTorch, labels must be LongTensor for CrossEntropyLoss.

Learning rate too high: Loss nan/oscillation → reduce LR (e.g., 1e-3 → 5e-4).

For TF: ensure model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy']) with from_logits=True if last layer has no softmax.

---

## Task 2 — MNIST CNN + Streamlit

Train notebook: `task2/deeplearning.ipynb`  
Weights: `task2/models/mnist_cnn.pt`  
App: `task2/app.py`

**Run locally**

```bash
pip install -r requirements.txt
cd task2
streamlit run app.py
```
streamlit deployment link, https://fadhi1i-week3-task2app-fyr59f.streamlit.app/
