.PHONY: test

test:
	python test_data_gen.py | python smooth.py --degree=4 --scanprocs=16 --window=20 ${SMOOTH_ARGS}
