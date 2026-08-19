chestxray-mlops

An end-to-end MLOps pipeline for chest X-ray classification — from versioned data and reproducible training through experiment tracking, containerisation and continuous integration.

The purpose of this repository is the pipeline, not the model: it demonstrates how a medical-imaging experiment can be made reproducible, auditable and portable.

Author: Sumaira Tabassum Awan

Pipeline
Stage	Tooling
Data versioning	DVC — datasets and intermediate artefacts tracked outside git, pipeline stages defined declaratively
Training	PyTorch — [model architecture] trained on chest radiographs for [task, e.g. pneumonia vs normal classification]
Experiment tracking	MLflow — parameters, metrics and artefacts logged per run
Containerisation	Docker — pinned environment, reproducible on any host
Continuous integration	GitHub Actions — pipeline runs automatically on push
Repository structure
src/                  training, evaluation and data-preparation code
.dvc/                 DVC configuration and pipeline stage definitions
.github/workflows/    CI pipeline definition
Dockerfile            container image build
requirements.txt      pinned Python dependencies
Running it

With Docker (recommended — no local Python setup required):

bash
docker build -t chestxray-mlops .
docker run --rm chestxray-mlops

Locally:

bash
pip install -r requirements.txt
dvc repro          # reproduce the full pipeline
mlflow ui          # inspect logged runs at http://localhost:5000

dvc repro re-executes only the stages whose inputs have changed, so re-running after a code change rebuilds the minimum necessary.

Data

Chest X-Ray Images (Pneumonia) dataset (Kermany et al., 2018), available on Kaggle under CC BY 4.0. Images are publicly available and anonymised; no identifiable patient information is included.

Data is not committed to this repository — it is tracked with DVC and pulled separately.

Scope and limitations

This is a research and engineering demonstration.

The model is not clinically validated and has no regulatory approval under the EU Medical Device Regulation or the EU AI Act.
It is not suitable for diagnostic use or clinical decision-making. 
Licence

[Licence, e.g. MIT]

Contact

Sumaira Tabassum Awan , ORCID 0009-0004-5637-8173
