# Stock-Predictor-ML-Inference

The objective of this code is to create a model inference server via Flask to send stock predictions from the trained ML model to the frontend and monitor model and data drift using Evidently AI.

## Setup

To run this code locally you can either install via conda:

`conda env create -f environment.yml`

Or through pip using the two commands below:

`pip install requirements.txt`

`pip install -e .`

## Deployment

This backend is hosted on Render.com for free.

## Connected Services

This repository interacts with the following services below:

<ol>
<li><a href="https://github.com/marcus-24/Stock-Predictor-ML-Training">Stock-Predictor-ML-Training</a></li>
<li><a href="https://github.com/marcus-24/Stock-Predictor-Frontend">Stock-Predictor-Frontend</a></li>
</ol>
