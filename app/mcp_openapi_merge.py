from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from contextlib import asynccontextmanager
import inspect


def _mcp_server_tools(server) -> Dict[str, Any]:
    """
    Get tools from tool_manager.
    Expected return: {tool_name: tool_obj}
    """

    tool_manager = server._tool_manager
    out = {}
    if tool_manager:
        for t in tool_manager.list_tools():
            out[t.name] = t
    return out


def _mcp_server_resources(server) -> List[Dict[str, Any]]:
    """
    Returns a list of resources from resources_manager
    """

    resource_manager = server._resource_manager
    out = {}

    if resource_manager:
        for t in resource_manager.list_resources():
            out[t.name] = t
    return out


def _mcp_server_resource_templates(server) -> List[Dict[str, Any]]:
    """
    Returns a list of resource templates from resources_manager
    """

    resource_manager = server._resource_manager
    out = {}

    if resource_manager:
        for t in resource_manager.list_templates():
            out[t.name] = t
    return out


def build_mcp_openapi_dict(
    server: Any,
    *,
    title: str = "MCP Server (Virtual HTTP for Docs)",
    version: str = "1.0.0",
    prefix: str = "/api/v1/mcp",
) -> Dict[str, Any]:
    """
    Build an OpenAPI for a "virtual endpoint" based on the FastMCP server's registration information.
    - Each tool => POST {prefix}/tools/{tool_name}

    The requestBody uses the tool's own schema (if missing, the loose schema is used).
    - Resource list => GET {prefix}/resources
    - SSE base endpoint (optional description) => GET {prefix}/sse
    """
    tools = _mcp_server_tools(server)
    resources = _mcp_server_resources(server)
    resource_templates = _mcp_server_resource_templates(server)

    openapi: Dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {"title": title, "version": version},
        "paths": {},
        "components": {"schemas": {}},
    }

    openapi.setdefault("components", {}).setdefault("schemas", {})

    # Tools: Mapped to POST /tools/{name}
    for name, tool in tools.items():
        input_schema = getattr(tool, "parameters", None)
        output_schema = getattr(
            getattr(tool, "fn_metadata", None), "output_schema", None
        )

        path = f"{prefix}/tools/{name}"

        post_obj = {
            "summary": f"Call MCP Tool: {name}",
            "description": (
                f"""{tool.description}

Virtual HTTP wrapper: Request bodies calling MCP tool `{name}`
will be forwarded to the MCP channel.
"""
            ),
            "tags": ["mcp-tools"],
            "responses": {
                "200": {
                    "description": "Tool return result (transmitted MCP response result as is)"
                }
            },
        }

        # input schema
        if input_schema:
            schema_name = f"{name}Input"
            openapi["components"]["schemas"][schema_name] = input_schema
            post_obj["requestBody"] = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": f"#/components/schemas/{schema_name}"}
                    }
                },
            }

        # output schema
        if output_schema:
            schema_name = f"{name}Output"
            openapi["components"]["schemas"][schema_name] = output_schema
            post_obj["responses"]["200"]["content"] = {
                "application/json": {
                    "schema": {"$ref": f"#/components/schemas/{schema_name}"}
                }
            }

        openapi["paths"][path] = {"post": post_obj}

    # Resource List
    res_path = f"{prefix}/resources"
    openapi["paths"][res_path] = {
        "get": {
            "summary": "List MCP Resources",
            "responses": {
                "200": {
                    "description": "Resource List",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "uri": {"type": "string"},
                                        "name": {"type": "string"},
                                        "description": {"type": "string"},
                                        "mimeTypes": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": ["uri"],
                                },
                            }
                        }
                    },
                }
            },
            "tags": ["mcp-resources"],
        }
    }

    sse_path = f"{prefix}/sse"
    openapi["paths"][sse_path] = {
        "get": {
            "summary": "MCP SSE Endpoint",
            "description": "Endpoint for establishing an SSE channel with the MCP client; tool calls are transmitted over this connection via JSON-RPC.",
            "responses": {"200": {"description": "Event stream"}},
            "tags": ["mcp-core"],
        }
    }

    return openapi


def merge_openapi_into_app(app: FastAPI, sub_openapi: Dict[str, Any]) -> None:
    """
    Merge sub_openapi's paths/components into app's openapi_schema.
    """
    main_schema = app.openapi()
    # merge paths
    for p, item in sub_openapi.get("paths", {}).items():
        main_schema.setdefault("paths", {})[p] = item
    # merge components
    sub_comp = sub_openapi.get("components") or {}
    main_comp = main_schema.setdefault("components", {})
    for comp_type, comp_dict in sub_comp.items():
        dst = main_comp.setdefault(comp_type, {})
        dst.update(comp_dict)
    app.openapi_schema = main_schema
