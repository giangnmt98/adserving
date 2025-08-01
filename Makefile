SHELL = /bin/bash
PYTHON := python3
VENV_NAME = adserving_env
MAIN_FOLDER = adserving
TEST_FOLDER = tests

# Environment
venv:
	${PYTHON} -m venv ${VENV_NAME} && \
	source ${VENV_NAME}/bin/activate && \
	${PYTHON} -m pip install pip setuptools wheel && \
	${PYTHON} -m pip install -e .[dev] && \
	pre-commit install

# Style
style:
	black ./${MAIN_FOLDER}/
	flake8 ./${MAIN_FOLDER}/
	${PYTHON} -m isort -rc ./${MAIN_FOLDER}/

test:
	${PYTHON} -m flake8 ./${MAIN_FOLDER}/
	${PYTHON} -m mypy ./${MAIN_FOLDER}/
# 	CUDA_VISIBLE_DEVICES=""  ${PYTHON} -m pytest -s --durations=0 --disable-warnings ${TEST_FOLDER}/
	${PYTHON} -m pylint ./${MAIN_FOLDER}/
