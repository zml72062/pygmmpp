all:
	cd ..; python3 pygmmpp/setup.py install --user

test:
	cd ..; pytest

clean:
	find . | grep -E "(/__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	rm -rf .pytest_cache
	rm -rf ../build