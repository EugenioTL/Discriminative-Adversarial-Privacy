# Discriminative Adversarial Privacy: Balancing Accuracy and Membership Privacy in Neural Networks (BMVC2023)

<!-- [![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2306.03054) -->
[![Paper](https://img.shields.io/badge/arXiv-Paper-blue.svg)](https://arxiv.org/abs/2306.03054)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

The official TensorFlow implementation of the BMVC2023 paper [**Discriminative Adversarial Privacy: Balancing Accuracy and Membership Privacy in Neural Networks**](https://arxiv.org/abs/2306.03054)  

Authors: Eugenio Lomurno, Alberto Archetti, Francesca Ausonio, Matteo Matteucci

## Introduction
This repository contains the TensorFlow implementation for "Discriminative Adversarial Privacy," a novel approach to balancing model accuracy and membership privacy in neural networks. The implementation showcases how to protect data privacy effectively without significantly compromising model performance.

## Installation
To get started with DAP, follow these steps:

1. **Clone the Repository**:

```bash
git clone https://github.com/EugenioTL/discriminative-adversarial-privacy
```

2. **Install Dependencies**:
Navigate to the project directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage
- **Jupyter Notebook Demo**: 
'Discriminative Adversarial Privacy - Privacy Preserving Component Demo.ipynb' provides a comprehensive demonstration of DAP's capabilities. It's an excellent starting point for understanding the project's functionality.

- **Running Scripts**:
The 'scripts' directory contains various scripts for tasks such as data preprocessing, model training, and evaluation. These scripts are vital for experimenting with and implementing the DAP framework in your projects.
- `privacy.py`: Contains the main logic for privacy-preserving mechanisms.
- `training.py`: Used for training the models with privacy considerations.

- **Model Directory**:
The 'models' directory houses the machine learning models used in DAP. These models are central to the framework's functionality and can be customized as per specific requirements.

## Docker Deployment
For ease of deployment, especially in isolated environments, DAP can be containerized using Docker:

1. **Build the Docker Image**:

```bash
docker build -t dap-image .
```

2. **Run the Container**:

```bash
docker run -it dap-image
```

## Citation
Should you find this repository useful, please consider citing:
```bibtex
@article{lomurno2023discriminative,
  title={Discriminative Adversarial Privacy: Balancing Accuracy and Membership Privacy in Neural Networks},
  author={Lomurno, Eugenio and Archetti, Alberto and Ausonio, Francesca and Matteucci, Matteo},
  journal={arXiv preprint arXiv:2306.03054},
  year={2023}
}
```

## License
This project is licensed under the terms of the [LICENSE](LICENSE) file. Please refer to the file for full licensing details.

## Contact Information
For further inquiries or collaboration opportunities, feel free to contact the authors or raise an issue in this repository.