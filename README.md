# machine-learning
Playground for Machine Learning and MLOps


# Target Architecture

```
machine-learning/
├── .github/
├── .dvc/
├── docs/
├── platform/
│   ├── api/
│   ├── orchestrator/
│   ├── registry/
│   ├── deployment/
│   └── monitoring/
├── src/
│   ├── common/
│   │   ├── logging/
│   │   ├── metrics/
│   │   ├── preprocessing/
│   │   ├── validation/
│   │   ├── exceptions/
│   │   └── utils/
│   │
│   ├── model1/
│   └── model2/
│
├── tests/
└── pyproject.toml
```
## Platform Architecture

The repository is organized around a reusable Machine Learning Platform rather than individual ML projects.

The `ml_platform` package contains reusable infrastructure that will be shared across future machine learning applications.

Current platform modules include:

- Configuration
- Data
- Models
- Training
- Evaluation
- Utilities

Additional platform capabilities will be introduced incrementally throughout the project roadmap.
