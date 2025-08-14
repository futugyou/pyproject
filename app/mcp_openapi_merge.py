from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from contextlib import asynccontextmanager
import inspect


def _guess_tools(server) -> Dict[str, Any]:
    """
    Get tools from FastMCP/variants whenever possible.
    Expected return: {tool_name: tool_obj}
    """
    cand_attrs = [
        "tools",
        "_tools",
        "registered_tools",
        "tool_registry",
        "registry",
    ]
    for attr in cand_attrs:
        obj = getattr(server, attr, None)
        if obj:
            print(attr)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list):
                out = {}
                for t in obj:
                    name = (
                        getattr(t, "name", None)
                        or getattr(t, "__name__", None)
                        or getattr(getattr(t, "func", None), "__name__", None)
                    )
                    if name:
                        out[name] = t
                if out:
                    return out
            for sub in ("tools", "_tools", "items", "data"):
                maybe = getattr(obj, sub, None)
                if isinstance(maybe, dict):
                    return maybe

    d = getattr(server, "__dict__", {})
    for k, v in d.items():
        if isinstance(v, dict):
            if all(isinstance(_k, str) for _k in v.keys()):
                return v
    return {}


def _guess_tool_manager(server) -> Dict[str, Any]:
    """
    Get tools from tool_manager.
    Expected return: {tool_name: tool_obj}
    """

    tool_manager = server._tool_manager
    if tool_manager:
        out = {}
        for t in tool_manager.list_tools():
            name = (
                getattr(t, "name", None)
                or getattr(t, "__name__", None)
                or getattr(getattr(t, "func", None), "__name__", None)
            )
            if name:
                out[name] = t
        if out:
            return out
    return {}


def _guess_tool_schema(tool_obj) -> Optional[Dict[str, Any]]:
    """
    Generate a JSON Schema for a single tool.
    Prefer using the tool's own schema. Alternatively, use inspect.signature to create a loose schema.
    """
    for attr in ("schema", "input_schema", "parameters"):
        js = getattr(tool_obj, attr, None)
        if isinstance(js, dict):
            return js

    for attr in ("to_dict", "model_dump"):
        fn = getattr(tool_obj, attr, None)
        if callable(fn):
            try:
                d = fn()
                for key in ("schema", "input_schema", "parameters"):
                    js = d.get(key)
                    if isinstance(js, dict):
                        return js
            except Exception:
                pass

    # fallback: Rough inference via function signatures
    func = (
        getattr(tool_obj, "func", None)
        or getattr(tool_obj, "__call__", None)
        or tool_obj
    )
    if callable(func):
        try:
            sig = inspect.signature(func)
            props = {}
            required = []
            for name, p in sig.parameters.items():
                if name in ("self", "context", "ctx"):
                    continue
                ann = p.annotation
                t = "string"
                if ann in (int, "int"):
                    t = "integer"
                elif ann in (float, "float"):
                    t = "number"
                elif ann in (bool, "bool"):
                    t = "boolean"
                elif ann in (dict, "dict"):
                    t = "object"
                elif ann in (list, "list"):
                    t = "array"
                props[name] = {"type": t}
                if p.default is inspect._empty:
                    required.append(name)
            schema = {
                "type": "object",
                "properties": props,
                "additionalProperties": True,
            }
            if required:
                schema["required"] = required
            return schema
        except Exception:
            pass

    return {"type": "object", "additionalProperties": True}


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
    """
    tools = _guess_tools(server)
    managerTools = _guess_tool_manager(server)
    tools |= managerTools

    openapi: Dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {"title": title, "version": version},
        "paths": {},
        "components": {"schemas": {}},
    }

    # Tools: Mapped to POST /tools/{name}
    for name, tool in tools.items():
        schema = _guess_tool_schema(tool) or {"type": "object"}
        path = f"{prefix}/tools/{name}"
        openapi["paths"][path] = {
            "post": {
                "summary": f"Call MCP Tool: {name}",
                "description": f"Virtual HTTP wrapper: Request bodies calling MCP tool `{name}` will be forwarded to the MCP channel.",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": schema}},
                },
                "responses": {
                    "200": {
                        "description": "Tool return result (transmitted MCP response result as is)",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
                "tags": ["mcp-tools"],
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
