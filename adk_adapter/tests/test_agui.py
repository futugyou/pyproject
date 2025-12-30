"""Unit tests for agui.py module."""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import pytest
from fastapi import FastAPI

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from adk_adapter import agui


class TestRegisterADKAgents:
    """Test suite for register_adk_agents function."""

    @patch("adk_adapter.agui.add_adk_fastapi_endpoint")
    @patch("adk_adapter.agui.build_assistant_adk_agent")
    @patch("adk_adapter.agui.build_assistant_agent")
    @patch("adk_adapter.agui.client_factory")
    @patch("adk_adapter.agui.ADKAgent")
    def test_register_adk_agents_calls_build_llm(
        self,
        mock_adk_agent,
        mock_client_factory,
        mock_build_assistant,
        mock_build_adk_agent,
        mock_add_endpoint,
    ):
        """Test that register_adk_agents calls build_llm."""
        mock_llm = Mock()
        mock_client_factory.build_llm.return_value = mock_llm
        mock_base_agent = Mock()
        mock_build_assistant.return_value = mock_base_agent
        mock_adk_agent_instance = Mock()
        mock_build_adk_agent.return_value = mock_adk_agent_instance

        app = FastAPI()
        agui.register_adk_agents(app)

        # Verify build_llm was called
        mock_client_factory.build_llm.assert_called_once()

    @patch("adk_adapter.agui.add_adk_fastapi_endpoint")
    @patch("adk_adapter.agui.build_assistant_adk_agent")
    @patch("adk_adapter.agui.build_assistant_agent")
    @patch("adk_adapter.agui.client_factory")
    def test_register_adk_agents_calls_build_assistant_agent(
        self,
        mock_client_factory,
        mock_build_assistant,
        mock_build_adk_agent,
        mock_add_endpoint,
    ):
        """Test that register_adk_agents calls build_assistant_agent with LLM."""
        mock_llm = Mock()
        mock_client_factory.build_llm.return_value = mock_llm
        mock_base_agent = Mock()
        mock_build_assistant.return_value = mock_base_agent
        mock_adk_agent_instance = Mock()
        mock_build_adk_agent.return_value = mock_adk_agent_instance

        app = FastAPI()
        agui.register_adk_agents(app)

        # Verify build_assistant_agent was called with the LLM
        mock_build_assistant.assert_called_once_with(mock_llm)

    @patch("adk_adapter.agui.add_adk_fastapi_endpoint")
    @patch("adk_adapter.agui.build_assistant_adk_agent")
    @patch("adk_adapter.agui.build_assistant_agent")
    @patch("adk_adapter.agui.client_factory")
    def test_register_adk_agents_calls_build_assistant_adk_agent(
        self,
        mock_client_factory,
        mock_build_assistant,
        mock_build_adk_agent,
        mock_add_endpoint,
    ):
        """Test that register_adk_agents calls build_assistant_adk_agent."""
        mock_llm = Mock()
        mock_client_factory.build_llm.return_value = mock_llm
        mock_base_agent = Mock()
        mock_build_assistant.return_value = mock_base_agent
        mock_adk_agent_instance = Mock()
        mock_build_adk_agent.return_value = mock_adk_agent_instance

        app = FastAPI()
        agui.register_adk_agents(app)

        # Verify build_assistant_adk_agent was called with base_agent
        mock_build_adk_agent.assert_called_once_with(mock_base_agent)

    @patch("adk_adapter.agui.add_adk_fastapi_endpoint")
    @patch("adk_adapter.agui.build_assistant_adk_agent")
    @patch("adk_adapter.agui.build_assistant_agent")
    @patch("adk_adapter.agui.client_factory")
    def test_register_adk_agents_calls_add_adk_fastapi_endpoint(
        self,
        mock_client_factory,
        mock_build_assistant,
        mock_build_adk_agent,
        mock_add_endpoint,
    ):
        """Test that register_adk_agents calls add_adk_fastapi_endpoint with correct parameters."""
        mock_llm = Mock()
        mock_client_factory.build_llm.return_value = mock_llm
        mock_base_agent = Mock()
        mock_build_assistant.return_value = mock_base_agent
        mock_adk_agent_instance = Mock()
        mock_build_adk_agent.return_value = mock_adk_agent_instance

        app = FastAPI()
        agui.register_adk_agents(app)

        # Verify add_adk_fastapi_endpoint was called with correct parameters
        mock_add_endpoint.assert_called_once_with(
            app=app, agent=mock_adk_agent_instance, path="/adk_assistant"
        )

    @patch("adk_adapter.agui.add_adk_fastapi_endpoint")
    @patch("adk_adapter.agui.build_assistant_adk_agent")
    @patch("adk_adapter.agui.build_assistant_agent")
    @patch("adk_adapter.agui.client_factory")
    def test_register_adk_agents_uses_correct_path(
        self,
        mock_client_factory,
        mock_build_assistant,
        mock_build_adk_agent,
        mock_add_endpoint,
    ):
        """Test that register_adk_agents uses the correct endpoint path."""
        mock_llm = Mock()
        mock_client_factory.build_llm.return_value = mock_llm
        mock_base_agent = Mock()
        mock_build_assistant.return_value = mock_base_agent
        mock_adk_agent_instance = Mock()
        mock_build_adk_agent.return_value = mock_adk_agent_instance

        app = FastAPI()
        agui.register_adk_agents(app)

        # Verify the path is correct
        call_kwargs = mock_add_endpoint.call_args[1]
        assert call_kwargs["path"] == "/adk_assistant"

    @patch("adk_adapter.agui.add_adk_fastapi_endpoint")
    @patch("adk_adapter.agui.build_assistant_adk_agent")
    @patch("adk_adapter.agui.build_assistant_agent")
    @patch("adk_adapter.agui.client_factory")
    def test_register_adk_agents_with_different_app(
        self,
        mock_client_factory,
        mock_build_assistant,
        mock_build_adk_agent,
        mock_add_endpoint,
    ):
        """Test that register_adk_agents can be called with different FastAPI apps."""
        mock_llm = Mock()
        mock_client_factory.build_llm.return_value = mock_llm
        mock_base_agent = Mock()
        mock_build_assistant.return_value = mock_base_agent
        mock_adk_agent_instance = Mock()
        mock_build_adk_agent.return_value = mock_adk_agent_instance

        app1 = FastAPI()
        app2 = FastAPI()

        agui.register_adk_agents(app1)
        agui.register_adk_agents(app2)

        # Verify both apps were used
        calls = mock_add_endpoint.call_args_list
        assert calls[0][1]["app"] == app1
        assert calls[1][1]["app"] == app2
        assert mock_add_endpoint.call_count == 2

    @patch("adk_adapter.agui.add_adk_fastapi_endpoint")
    @patch("adk_adapter.agui.build_assistant_adk_agent")
    @patch("adk_adapter.agui.build_assistant_agent")
    @patch("adk_adapter.agui.client_factory")
    def test_register_adk_agents_call_order(
        self,
        mock_client_factory,
        mock_build_assistant,
        mock_build_adk_agent,
        mock_add_endpoint,
    ):
        """Test that functions are called in the correct order."""
        mock_llm = Mock()
        mock_client_factory.build_llm.return_value = mock_llm
        mock_base_agent = Mock()
        mock_build_assistant.return_value = mock_base_agent
        mock_adk_agent_instance = Mock()
        mock_build_adk_agent.return_value = mock_adk_agent_instance

        app = FastAPI()
        agui.register_adk_agents(app)

        # Verify call order: build_llm -> build_assistant_agent -> build_assistant_adk_agent -> add_adk_fastapi_endpoint
        assert mock_client_factory.build_llm.called
        assert mock_build_assistant.called
        assert mock_build_adk_agent.called
        assert mock_add_endpoint.called

        # Check that build_llm was called before build_assistant_agent
        assert (
            mock_client_factory.build_llm.call_index < mock_build_assistant.call_index
        )

    @patch("adk_adapter.agui.add_adk_fastapi_endpoint")
    @patch("adk_adapter.agui.build_assistant_adk_agent")
    @patch("adk_adapter.agui.build_assistant_agent")
    @patch("adk_adapter.agui.client_factory")
    def test_register_adk_agents_passes_agent_correctly(
        self,
        mock_client_factory,
        mock_build_assistant,
        mock_build_adk_agent,
        mock_add_endpoint,
    ):
        """Test that the agent is correctly passed through the chain."""
        mock_llm = Mock()
        mock_client_factory.build_llm.return_value = mock_llm
        mock_base_agent = Mock()
        mock_build_assistant.return_value = mock_base_agent
        mock_adk_agent_instance = Mock()
        mock_build_adk_agent.return_value = mock_adk_agent_instance

        app = FastAPI()
        agui.register_adk_agents(app)

        # Verify that base_agent from build_assistant_agent is passed to build_assistant_adk_agent
        build_adk_call_kwargs = mock_build_adk_agent.call_args[1]
        assert (
            build_adk_call_kwargs[0] == mock_base_agent
            if build_adk_call_kwargs
            else build_adk_call_kwargs
        )

        # Verify that adk_agent from build_assistant_adk_agent is passed to add_adk_fastapi_endpoint
        add_endpoint_call_kwargs = mock_add_endpoint.call_args[1]
        assert add_endpoint_call_kwargs["agent"] == mock_adk_agent_instance

    @patch("adk_adapter.agui.add_adk_fastapi_endpoint")
    @patch("adk_adapter.agui.build_assistant_adk_agent")
    @patch("adk_adapter.agui.build_assistant_agent")
    @patch("adk_adapter.agui.client_factory")
    def test_register_adk_agents_no_return_value(
        self,
        mock_client_factory,
        mock_build_assistant,
        mock_build_adk_agent,
        mock_add_endpoint,
    ):
        """Test that register_adk_agents doesn't return a value."""
        mock_llm = Mock()
        mock_client_factory.build_llm.return_value = mock_llm
        mock_base_agent = Mock()
        mock_build_assistant.return_value = mock_base_agent
        mock_adk_agent_instance = Mock()
        mock_build_adk_agent.return_value = mock_adk_agent_instance

        app = FastAPI()
        result = agui.register_adk_agents(app)

        # The function should not return anything
        assert result is None


class TestAguiImports:
    """Test suite for agui module imports."""

    def test_agui_imports_adkagent(self):
        """Test that agui module imports from adk_adapter.adkagent.assistant."""
        from adk_adapter import agui

        # Verify that the functions are available
        assert hasattr(agui, "register_adk_agents")
        assert callable(agui.register_adk_agents)

    def test_agui_imports_from_ag_ui_adk(self):
        """Test that agui module imports from ag_ui_adk."""
        from adk_adapter import agui

        # ag_ui_adk imports should be available at module level
        # (The actual imports are at the top of agui.py)
        assert True  # If we got here without import error, imports work
