.PHONY: test docs

tests:
	python -m unittest discover test/

clean:
	rm -rf .cache/
	rm -rf *.egg-info
	rm -rf build/
	rm -rf dist/

docs:
	cd docs && $(MAKE) html

publish: clean
	python setup.py sdist bdist bdist_wheel
	twine upload dist/*
	@$(MAKE) clean
