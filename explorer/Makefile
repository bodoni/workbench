all: data

data: assets/png/.done

assets/png/.done: assets/svg/.done
	mkdir -p assets/png
	qlmanage -t -s 224 -o assets/png assets/svg/*
	touch $@

assets/svg/.done:
	mkdir -p assets/svg
	cargo run --bin founder-sign -- \
		--path ../founder/tests/fixtures/google-fonts \
		--output assets/svg \
		--characters anop \
		--ignore google-fonts/ofl/bungeecolor \
		--ignore google-fonts/ofl/bungeespice \
		--ignore google-fonts/ofl/gruppo \
		--ignore google-fonts/ofl/iceland \
		--ignore google-fonts/ofl/kaushanscript \
		--ignore google-fonts/ufl/ubuntu \
		--workers 4
	touch $@

.PHONY: all data
