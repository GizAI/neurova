SHELL := /bin/bash
.PHONY: source-test verify hdl clean package
source-test:
	./scripts/run_source_tests.sh
hdl:
	./scripts/run_hdl_tests.sh
verify: source-test hdl
clean:
	rm -rf build/* reports/*.log reports/*.status
package:
	python3 tools/package_release.py
