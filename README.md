# Discriminative Adversarial Privacy: Balancing Accuracy and Membership Privacy in Neural Networks (BMVC2023)

[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2306.03054)

The official TensorFlow implementation of the BMVC2023 paper [**Discriminative Adversarial Privacy: Balancing Accuracy and Membership Privacy in Neural Networks**](https://arxiv.org/abs/2306.03054)  

Authors: Eugenio Lomurno, Alberto Archetti, Francesca Ausonio, Matteo Matteucci

## Introduction
This repository contains the TensorFlow implementation for "Discriminative Adversarial Privacy," a novel approach to balancing model accuracy and membership privacy in neural networks. The implementation showcases how to protect data privacy effectively without significantly compromising model performance.

## Installation and Requirements
To set up your environment for using this code, follow these steps:

1. Install the required libraries using `requirements.txt`:

```bash
pip install -r requirements.txt
```

2. For Docker users, a `Dockerfile` is provided for containerized setup.

## Usage
The repository includes several scripts and a Jupyter Notebook for demonstration:
- `privacy.py`: Contains the main logic for privacy-preserving mechanisms.
- `training.py`: Used for training the models with privacy considerations.
- The Jupyter Notebook (`Discriminative Adversarial Privacy - Privacy Preserving Component Demo.ipynb`) provides a practical demo of how to apply these concepts.


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
This project is licensed under the terms of the MIT License. See the `LICENSE` file for more details.

## Contact Information
For further inquiries or collaboration opportunities, feel free to contact the authors or raise an issue in this repository.