# NOTE: `make` is not installed on the primary Windows dev machine this was built on (verified:
# `make --version` -> command not found). These targets are the documented reproduction commands
# for any environment that does have `make` (Linux/macOS/WSL/CI); on the Windows box, run the
# `py -m ...` command from the right-hand side of each target directly. See CLAUDE.md.

.PHONY: eval eval-fast eval-holdout test smoke

eval:
	py -m eval.run_eval --mode full

eval-fast:
	py -m eval.run_eval --mode fast

eval-holdout:
	py -m eval.run_eval --mode holdout

test:
	py -m unittest discover -s tests -v

smoke:
	py -m unittest discover -s tests -p "test_failure_contract.py" -v
