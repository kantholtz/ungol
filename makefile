#
#   VNGOL PYTHON MODULE MANAGEMENT
#


.PHONY: setup dist clean


INST := pip install
PY := python3

dir_base := ungol

dir_common := common
dir_models := models
dir_similarity := similarity
dir_eval := eval

lns := \
	$(dir_base)/common \
	$(dir_base)/models \
	$(dir_base)/index \
	$(dir_base)/similarity \
	$(dir_base)/retrieval \
	$(dir_base)/sentemb


#
# --------------------
#

setup: $(dir_base) $(lns)
	$(INST) -r $(dir_common)/requirements.txt
	$(INST) -r $(dir_models)/requirements.txt
	$(INST) -r $(dir_similarity)/requirements.txt
	$(INST) -r $(dir_eval)/requirements.txt
	pip install -e .


dist: setup
	$(PY) -m pip install --upgrade setuptools wheel
	$(PY) setup.py sdist bdist_wheel

$(dir_base):
	@echo 'creating dir_base'
	mkdir $(dir_base)

$(dir_base)/common:
	ln -s ../$(dir_common)/ungol/common $(dir_base)/

$(dir_base)/models:
	ln -s ../$(dir_models)/ungol/models $(dir_base)/

$(dir_base)/sentemb:
	ln -s ../$(dir_models)/ungol/sentemb $(dir_base)/

$(dir_base)/index:
	ln -s ../$(dir_similarity)/ungol/index $(dir_base)/

$(dir_base)/similarity:
	ln -s ../$(dir_similarity)/ungol/similarity $(dir_base)/

$(dir_base)/retrieval:
	ln -s ../$(dir_eval)/ungol/retrieval $(dir_base)/


#
# --------------------
#

clean:
	rm -r $(dir_base) dist
