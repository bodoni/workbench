all: check

aim:
	aim up --port 6006

check:
	git ls-files -- . | xargs pre-commit run --files

data: assets/png/.done

assets/png/.done: assets/svg/.done
	mkdir -p assets/png
	if hash qlmanage; then \
		qlmanage -t -s 256 -o assets/png assets/svg/*; \
	else \
		for file in assets/svg/*; do \
			dbus-run-session inkscape -z -b ffffff -w 256 -h 256 $${file} -e $$(echo $${file} | sed -r "s#/svg#/png#g").png; \
		done \
	fi
	touch $@

assets/svg/.done:
	mkdir -p assets/svg
	cargo run --bin founder-sign -- \
		--path ../tests/fixtures/google-fonts \
		--characters anop \
		--output assets/svg \
		--ignore google-fonts/ofl/bungeecolor \
		--ignore google-fonts/ofl/bungeespice \
		--ignore google-fonts/ofl/gruppo \
		--ignore google-fonts/ofl/iceland \
		--ignore google-fonts/ofl/kaushanscript \
		--ignore google-fonts/ufl/ubuntu \
		--workers 4
	touch $@

.PHONY: all aim check data
