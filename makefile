# Makefile for managing virtual environments for sub-projects

# List of sub-projects
SUBPROJECTS = a2a_samples agent_adapter autogen_demo langchain_adapter mcp_adapter oauth_client public_opinion_monitoring semantic_kernel_adapter

# Create virtual environments and install dependencies
.PHONY: all create-venv install-deps

# Default target: create virtual environments and install
all: create-venv install-deps

# Create virtual environments for each sub-project
create-venv:
	@echo "Creating virtual environments for subprojects..."
	@for dir in $(SUBPROJECTS); do \
		if [ ! -d "$$dir/.venv" ]; then \
			echo "Creating virtual environment for $$dir..."; \
			python -m venv $$dir/.venv; \
		else \
			echo "Virtual environment for $$dir already exists."; \
		fi \
	done

# Install dependencies for each sub-project
install-deps:
	@echo "Installing dependencies for subprojects..."
	@for dir in $(SUBPROJECTS); do \
		if [ -d "$$dir/.venv" ]; then \
			echo "Installing dependencies for $$dir..."; \
			$$dir/.venv/bin/pip install -r $$dir/requirements.txt; \
		else \
			echo "No virtual environment found for $$dir!"; \
		fi \
	done

