.PHONY: test

test:
	python test_data_gen.py | python smooth.py --degree=4 --steps=1000 --perturb=100 --maxprocs=16 --window=20 ${SMOOTH_ARGS}
