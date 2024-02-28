from setuptools import setup, find_packages

setup(
    name='gemma_finetuning_project',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'datasets',
        'trl',
        'peft',
        'bitsandbytes',
        'huggingface_hub',
    ],
    python_requires='>=3.7',
    description='A project for fine-tuning the Gemma model.',
    author='Your Name',
    author_email='your.email@example.com',
    keywords='machine learning, fine-tuning, gemma',
)
