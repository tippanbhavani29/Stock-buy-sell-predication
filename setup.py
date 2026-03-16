from setuptools import find_packages, setup

setup(
    name="mlflow_stock_project",
    version="0.0.1",
    author="Bhavani",
    author_email="your_email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "mlflow",
        "xgboost",
        "yfinance",
        "matplotlib",
        "seaborn",
        "ta",
        "shap"
    ]
)
#this file makes  your project installable like a python pacakage
