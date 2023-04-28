.PHONY: clean
clean: clean-python clean-package clean-tests clean-system

.PHONY: clean-python
clean-python:
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*.pyd' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -rf {} +

.PHONY: clean-package
clean-package:
	@rm -rf dist/
	@rm -rf build/
	@rm -rf .eggs/
	@find . -name '*.egg-info' -exec rm -rf {} +
	@find . -name '*.egg' -exec rm -rf {} +

.PHONY: clean-tests
clean-tests:
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .tox/
	@rm -f .coverage
	@rm -rf htmlcov/

.PHONY: clean-system
clean-system:
	@find . -name '*~' -exec rm -f {} +
	@find . -name '.DS_Store' -exec rm -f {} +

.PHONY: requirements
requirements:
	poetry export --without-hashes -f requirements.txt -o requirements.txt

.PHONY: stubs
stubs:
	stubgen -o . -p class_activation_mapping

.PHONY: build-package
build-package:
	$(eval VERSION := $(shell poetry version -s))
	poetry build
	@tar zxf dist/class_activation_mapping-$(VERSION).tar.gz -C ./dist
	@cp dist/class_activation_mapping-$(VERSION)/setup.py setup.py
	@black setup.py
	@rm -rf dist

.PHONY: install
install:
	python setup.py install

.PHONY: uninstall
uninstall:
	pip uninstall -y class-activation-mapping

.PHONY: docs
docs:
	cd docs && make html

.PHONY: tests
tests: tests-python

.PHONY: tests-python
tests-python:
	poetry run pytest

.PHONY: tests-report
tests-report:
	python -u -m pytest -v --cov --cov-report=html

# add new version number.
# do this after committing changes to the local repositry
# and before pushing changes to the remote repository.
.PHONY: version-up
version-up:
ifdef VERSION
	git tag $(VERSION)
	poetry dynamic-versioning
	git add pyproject.toml class_activation_mapping/__init__.py
	git commit -m "$(VERSION)"
	git tag -f $(VERSION)
	git push
	git push --tags
else
	@echo "Usage: make version-up VERSION=vX.X.X"
endif
