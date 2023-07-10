# MetaBerta - Metagenomics Analysis with Customizable Language Models

MetaBerta is a project that enables metagenomics analysis using customizable language models. It provides a flexible pipeline that allows users to select their preferred architecture and language model (currently supporting BERT and Roberta) for training and analysis. The pipeline includes components for data preprocessing, model training, embedding generation, and evaluation.

## Pipeline Features

- **Customizable Language Models**: Users can choose between BERT and Roberta architectures as their language model for metagenomics analysis. This flexibility allows for fine-tuning or transfer learning based on specific requirements.

- **Training**: The pipeline supports training the selected language model on metagenomic data. Users can provide their training data and specify the necessary hyperparameters to train the model.

- **Embedding**: MetaBerta allows users to generate embeddings for metagenomic sequences using the trained language model. These embeddings capture the semantic information of the sequences, enabling downstream analysis.

- **Evaluation**: The pipeline provides evaluation functionalities to assess the performance of the trained model on metagenomic tasks. Users can evaluate their model using various metrics, analyze the results, and visualize the performance.
