.PHONY: clean requirements

###############################################################################
# GLOBALS                                                                     #
###############################################################################
SHELL := /bin/bash  # use bash syntax
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = radio
PYTHON_INTERPRETER = python3
USERNAME = eduardojdiniz
EMAIL = edd32@pitt.edu
NETRC ?= "True"
GITHUB_TOKEN_FILE ?=
GITHUB_TOKEN ?= "$(cat "$(GITHUB_TOKEN_FILE)")"

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

REGEX := "*'$(PROJECT_NAME)'*"
ENVS=$(shell conda env list | awk '{print $1}' )

ifeq ($(shell echo $(ENVS) | egrep "$(REGEX)"),)
HAS_ENV=False
else
HAS_ENV=True
endif

ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
PYTHON_VERSION="3.8"
else
PYTHON_VERSION="2.7"
endif

# When installing multiple packages in a script, the installation needs to be
# done as the root user. There are three general options that can be used to do
# this:

# 1. Run the entire script as the root user (not recommended).
# 2. Use the sudo command from the Sudo package.
# 3. Use su -c "command arguments" (quotes required) which will ask for the
# root password for every iteration of the loop.

# One way to handle this situation is to create a short bash function that
# automatically selects the appropriate method. Once the command is set in the
# environment, it does not need to be set again.

EUID := $(shell id -u -r)
define as_root
	@if [[ $(EUID) -eq 0 ]]; then $(1);\
	elif [[ -x /usr/bin/sudo ]]; then sudo $(1);\
	else su -c \\"$(1)\\";\
	fi;
endef

###############################################################################
# COMMANDS                                                                    #
###############################################################################

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "outputs" -exec rm -rf {} +
	find . -type d -name "multirun" -exec rm -rf {} +

## Set up Git
git:
	@echo "starting git repo..."
	@if [[ -d ".git" ]]; then \
		echo ".git already exists"; \
	else \
		git init; \
		git config user.name $(USERNAME); \
		git config user.email $(EMAIL); \
		git add --all && git commit -m "Initial commit, hello world!"; \
		git branch -m main; \
    fi;

## Set up GitHub
github: git
	@echo "adding github remote..."
	@if [[ -z $(GITHUB_TOKEN) && $(NETRC) != True ]]; then \
		echo "Necessary to pass file with your personal GitHub access token"; \
   		echo "as TOKEN_FILE=<path/to/file> or have the TOKEN environment"; \
	    echo "variable with your token string set upon calling this target"; \
		echo "Alternatively, setup a .netrc file at your home directory"; \
		echo "with the contents `machine github.com login yourusername password yourpassword`"; \
		echo ""; \
		echo "Please follow the instructions at https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token"; \
	elif [[ $(NETRC) = True ]]; then \
		curl --netrc https://api.github.com/user/repos \
			-d '{"name": "'$(PROJECT_NAME)'", "private": "true"}'; \
		git remote add origin git@github.com:$(USERNAME)/$(PROJECT_NAME).git; \
		echo "push main to origin"; \
		git push -u origin main; \
		echo "set origin as the default remote repo when on branch main"; \
		git config branch.main.remote origin; \
		echo "set origin main as the default remote branch when pull on the main branch"; \
		git config branch.main.merge refs/heads/main; \
		echo "push the current branch to a branch of the same name by default"; \
		git config push.default current; \
	else \
		curl -H "Authorization: token $(GITHUB_TOKEN)" https://api.github.com/user/repos \
			-d '{"name": "'$(PROJECT_NAME)'", "private": "false"}'; \
		git remote add origin git@github.com:$(USERNAME)/$(PROJECT_NAME).git; \
		echo "push main to origin"; \
		git push -u origin main; \
		echo "set origin as the default remote repo when on branch main"; \
		git config branch.main.remote origin; \
		echo "set origin main as the default remote branch when pull on the main branch"; \
		git config branch.main.merge refs/heads/main; \
		echo "push the current branch to a branch of the same name by default"; \
		git config push.default current; \
	fi;

## Set up python interpreter environment
virtual_environment:
	@if [[ $(HAS_CONDA) = False ]]; then \
		echo -e ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n"; \
		echo -e "export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"; \
		$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper; \
		bash `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); \
		echo -e ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"; \
	elif [[ $(HAS_ENV) = "False" ]]; then \
		echo -e ">>> Detected conda. Creating conda environment.\n>>> Activate environment with:\nconda activate $(PROJECT_NAME)"; \
		conda create -y -q --name $(PROJECT_NAME) python=$(PYTHON_VERSION); \
	else \
		echo -e ">>> Detected conda. Environment already exists.\n>>> Activate environment with:\nconda activate $(PROJECT_NAME)"; \
	fi;

## Install Python Dependencies
install_requirements: virtual_environment
	@if [[ $(HAS_CONDA) = True ]]; then \
		conda env update -q --name $(PROJECT_NAME) --file environment.yml --prune; \
	else \
		workon $(PROJECT_NAME); \
		$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel; \
		$(PYTHON_INTERPRETER) -m pip install -r requirements.txt; \
		$(PYTHON_INTERPRETER) -m pip install -r requirements-dev.txt; \
    fi;
	@echo "Add conda kernel to ipython"
	ipython kernel install --user --name=$(PROJECT_NAME)

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
# Test the entire {{ cookiecutter.repo_name }} module
test:
	py.test --pyargs $(PROJECT_NAME) --cov-report term-missing --cov=$(PROJECT_NAME)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
