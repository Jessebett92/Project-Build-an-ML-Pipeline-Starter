import pytest
import pandas as pd
import wandb
import os


def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")


@pytest.fixture(scope='session')
def wandb_run():
    run = wandb.init(job_type="data_tests", resume=True)
    yield run
    run.finish()

@pytest.fixture(scope='session')
def data(request, wandb_run):
    artifact_ref = request.config.option.csv
    if artifact_ref is None:   
        pytest.fail("You must provide the --csv option on the command line")

    artifact = wandb_run.use_artifact(artifact_ref)    
    artifact_path = artifact.download()
    filename = artifact_ref.split(":")[0]
    data_path = os.path.join(artifact_path, filename)

    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def ref_data(request, wandb_run):
    artifact_ref = request.config.option.ref
    if artifact_ref is None:
        pytest.fail("You must provide the --ref option on the command line")

    artifact = wandb_run.use_artifact(artifact_ref)        
    artifact_path = artifact.download()
    filename = artifact_ref.split(":")[0]
    data_path = os.path.join(artifact_path, filename)

    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)

@pytest.fixture(scope='session')
def min_price(request):
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)

@pytest.fixture(scope='session')
def max_price(request):
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)
