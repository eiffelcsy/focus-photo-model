"""
Setup configuration for Focus Photo Model - SmolVLM Photography Judging Pipeline
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="focus-photo-model",
    version="0.1.0",
    author="Focus Photo Model Team",
    description="SmolVLM-based photography judging pipeline with pseudolabel and self-supervised training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/focus-photo-model",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "mypy>=0.800",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "focus-train-supervised=scripts.train_supervised:main",
            "focus-train-self-supervised=scripts.train_self_supervised:main",
            "focus-train-hybrid=scripts.train_hybrid:main",
            "focus-generate-pseudolabels=scripts.generate_pseudolabels:main",
            "focus-evaluate=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
)
