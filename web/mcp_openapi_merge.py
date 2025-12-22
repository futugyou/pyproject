from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from contextlib import asynccontextmanager
import inspect
from pathlib import Path
from typing import Any
import re
import asyncio
from faker import Faker

fake = Faker("en_US")


def add_resource_properties(schema: dict, resource: Any, attrs: list):
    """
    Dynamically adds non-None properties from resource to schema['properties'].
    Automatically handles types:
    - str -> string
    - Path -> string
    - bytes -> string + format: binary
    """
    for attr in attrs:
        value = get_attr(resource, attr, None)
        if value is not None:
            prop_schema = {"example": None}
            if isinstance(value, bytes):
                prop_schema.update(
                    {
                        "type": "string",
                        "format": "binary",
                        "description": f"Binary content of {attr}",
                        "example": "<binary content omitted>",
                    }
                )
            elif isinstance(value, Path):
                prop_schema.update(
                    {
                        "type": "string",
                        "description": f"Path to {attr}",
                        "example": str(value),
                    }
                )
            else:
                prop_schema.update({"type": "string", "example": str(value)})
            schema["properties"][attr] = prop_schema


def get_attr(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


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


def _mcp_server_resources(server) -> Dict[str, Any]:
    """
    Returns a dict of resources from resources_manager
    """

    resource_manager = server._resource_manager
    out = {}

    if resource_manager:
        for t in resource_manager.list_resources():
            out[t.name] = t
    return out


def _mcp_server_prompts(server) -> Dict[str, Any]:
    """
    Returns a dict of prompts from _prompt_manager
    """

    prompt_manager = server._prompt_manager
    out = {}

    if prompt_manager:
        for t in prompt_manager.list_prompts():
            out[t.name] = t
    return out


def _mcp_server_resource_templates(server) -> Dict[str, Any]:
    """
    Returns a dict of resource templates from resources_manager
    """

    resource_manager = server._resource_manager
    out = {}

    if resource_manager:
        for t in resource_manager.list_templates():
            out[t.name] = t
    return out


def register_schema_recursive(openapi, schema, default_name, visited=None):
    if visited is None:
        visited = set()

    if not schema:
        return None

    if (
        not schema.get("properties")
        and not schema.get("additionalProperties")
        and "$ref" not in schema
    ):
        return None

    components = openapi.setdefault("components", {}).setdefault("schemas", {})

    schema_name = schema.get("title") or default_name
    if schema_name in visited:
        return {"$ref": f"#/components/schemas/{schema_name}"}
    visited.add(schema_name)

    defs = schema.get("$defs", {})
    for def_name, def_schema in defs.items():
        if def_name not in components:
            register_schema_recursive(openapi, def_schema, def_name, visited)

    def fix_refs(s):
        if isinstance(s, dict):
            new_s = {}
            for k, v in s.items():
                if k == "$ref" and isinstance(v, str) and v.startswith("#/$defs/"):
                    ref_name = v.split("/")[-1]
                    new_s[k] = f"#/components/schemas/{ref_name}"
                elif k in ("allOf", "anyOf", "oneOf") and isinstance(v, list):
                    new_s[k] = [fix_refs(item) for item in v]
                elif k == "not" and isinstance(v, dict):
                    new_s[k] = fix_refs(v)
                else:
                    new_s[k] = fix_refs(v)
            return new_s
        elif isinstance(s, list):
            return [fix_refs(item) for item in s]
        else:
            return s

    schema_clean = dict(schema)
    schema_clean.pop("$defs", None)
    schema_clean = fix_refs(schema_clean)

    if schema_name not in components:
        components[schema_name] = schema_clean

    return {"$ref": f"#/components/schemas/{schema_name}"}


def register_tool_schema(openapi, prefix, group, tool):
    name = tool.name
    path = f"{prefix}/tools/{name}"
    description = tool.description
    if tool.description is not None:
        description = description.rstrip()

    post_obj = {
        "summary": f"call {group} mcp tool: {name}",
        "description": f"""
        {description}

        Virtual HTTP wrapper: Request bodies calling MCP tool `{name}` will be forwarded to the MCP channel.
        """,
        "tags": [f"{group}_tools"],
        "responses": {"200": {"description": "Tool return result"}},
    }

    # input schema
    input_schema = getattr(tool, "parameters", None)
    input_ref = register_schema_recursive(openapi, input_schema, f"{name}Input")
    if input_ref:
        post_obj["requestBody"] = {
            "required": True,
            "content": {"application/json": {"schema": input_ref}},
        }

    # output schema
    output_schema = getattr(getattr(tool, "fn_metadata", None), "output_schema", None)
    output_ref = register_schema_recursive(openapi, output_schema, f"{name}Output")
    if output_ref:
        post_obj["responses"]["200"]["content"] = {
            "application/json": {"schema": output_ref}
        }

    openapi["paths"][path] = {"post": post_obj}


def register_resource_schema(openapi, prefix, group, resource):
    """
    Register an MCP resource into the OpenAPI spec for documentation purposes.
    Output schema will be registered with register_schema_recursive for consistency with tools.
    """
    name = resource.name
    uri_template = str(resource.uri)
    description = resource.description.rstrip()
    title = resource.title
    mime_type = resource.mime_type

    path = f"{prefix}/resources/{name}"
    path_params = re.findall(r"\{([^}]+)\}", uri_template)

    post_obj = {
        "summary": f"call {group} mcp resource: {name}",
        "description": f"""
        {description}

        Virtual HTTP wrapper: Request bodies calling MCP resource `{name}` will be forwarded to the MCP channel.
        uriTemplate: `{uri_template}`
        """,
        "tags": [f"{group}_resources"],
        "parameters": [],
        "responses": {"200": {"description": "Resource return result"}},
    }

    for p in path_params:
        post_obj["parameters"].append(
            {
                "name": p,
                "in": "path",
                "required": True,
                "schema": {"type": "string"},
                "description": f"{p} extracted from uriTemplate",
            }
        )

    output_schema = {
        "type": "object",
        "properties": {
            "uriTemplate": {"type": "string", "example": uri_template},
            "name": {"type": "string", "example": name},
            "title": {"type": "string", "example": title},
            "description": {"type": "string", "example": description},
            "mimeType": {"type": "string", "example": mime_type},
        },
        "required": ["uriTemplate", "name", "title", "description", "mimeType"],
    }

    attrs = ["path", "pattern", "text", "data"]
    add_resource_properties(output_schema, resource, attrs)

    output_ref = register_schema_recursive(
        openapi, output_schema, f"{name}ResourceOutput"
    )

    if output_ref:
        post_obj["responses"]["200"]["content"] = {mime_type: {"schema": output_ref}}

    openapi["paths"][path] = {"post": post_obj}


def register_resource_template_schema(openapi, prefix, group, template):
    """
    Register an MCP resource template into the OpenAPI spec for documentation purposes.
    Output schema will be registered with register_schema_recursive for consistency with tools.
    """
    uri_template = str(template.uri_template)
    name = template.name
    title = template.title
    description = template.description.rstrip()
    mime_type = template.mime_type

    path = f"{prefix}/templates/{name}"
    path_params = re.findall(r"\{([^}]+)\}", uri_template)

    post_obj = {
        "summary": f"call {group} mcp resource template: {name}",
        "description": f"""
        {description}

        Virtual HTTP wrapper: Request bodies calling MCP resource template `{name}` will be forwarded to the MCP channel.
        uriTemplate: `{uri_template}`
        """,
        "tags": [f"{group}_templates"],
        "parameters": [],
        "responses": {"200": {"description": "Resource template return result"}},
    }

    for p in path_params:
        post_obj["parameters"].append(
            {
                "name": p,
                "in": "path",
                "required": True,
                "schema": {"type": "string"},
                "description": f"{p} extracted from uriTemplate",
            }
        )

    # input schema
    input_schema = getattr(template, "parameters", None)
    input_ref = register_schema_recursive(openapi, input_schema, f"{name}Input")
    if input_ref:
        post_obj["requestBody"] = {
            "required": True,
            "content": {"application/json": {"schema": input_ref}},
        }

    output_schema = {
        "type": "object",
        "properties": {
            "uriTemplate": {"type": "string", "example": uri_template},
            "name": {"type": "string", "example": name},
            "title": {"type": "string", "example": title},
            "description": {"type": "string", "example": description},
            "mimeType": {"type": "string", "example": mime_type},
        },
        "required": ["uriTemplate", "name", "title", "description", "mimeType"],
    }

    output_ref = register_schema_recursive(
        openapi, output_schema, f"{name}ResourceTemplateOutput"
    )

    if output_ref:
        post_obj["responses"]["200"]["content"] = {mime_type: {"schema": output_ref}}

    openapi["paths"][path] = {"post": post_obj}


async def register_prompt_schema(openapi, prefix, group, prompt):
    """
    Register an MCP prompt into the OpenAPI spec for documentation purposes.
    Output schema will be registered with register_schema_recursive for consistency with tools.
    """
    name = prompt.name
    description = prompt.description.rstrip()
    title = prompt.title

    path = f"{prefix}/prompts/{name}"

    post_obj = {
        "summary": f"call {group} mcp prompt: {name}",
        "description": f"""
        {description}

        Virtual HTTP wrapper: Request bodies calling MCP prompt `{name}` will be forwarded to the MCP channel.
        """,
        "tags": [f"{group}_prompts"],
        "parameters": [],
        "responses": {"200": {"description": "Prompt return result"}},
    }

    fake_args = {}
    for p in prompt.arguments:
        if p.required:
            fake_args[p.name] = fake.word()
        post_obj["parameters"].append(
            {
                "name": p.name,
                "in": "path",
                "required": p.required,
                "schema": {"type": "string"},
                "description": p.name,
            }
        )

    prompt_result = ""
    try:
        messages = await prompt.render(fake_args)
        for message in messages:
            text = get_attr(message.content, "text", None)
            if text is not None:
                prompt_result += text + " "
    finally:
        pass
    output_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "example": prompt_result},
        },
        "required": ["prompt"],
    }

    output_ref = register_schema_recursive(
        openapi, output_schema, f"{name}PromptOutput"
    )

    if output_ref:
        post_obj["responses"]["200"]["content"] = {
            "application/json": {"schema": output_ref}
        }

    openapi["paths"][path] = {"post": post_obj}


async def build_mcp_openapi_dict(
    server: Any,
    *,
    title: str = "MCP Server (Virtual HTTP for Docs)",
    version: str = "1.0.0",
    prefix: str = "/api/v1/mcp",
    group: str = "mcp",
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
    prompts = _mcp_server_prompts(server)

    openapi: Dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {"title": title, "version": version},
        "paths": {},
        "components": {"schemas": {}},
    }

    # ----------------- create tools paths -----------------
    for name, tool in tools.items():
        register_tool_schema(openapi, prefix, group, tool)

    # ----------------- create resources paths -----------------
    for name, resource in resources.items():
        register_resource_schema(openapi, prefix, group, resource)

    # ----------------- create resource_templates paths -----------------
    for name, template in resource_templates.items():
        register_resource_template_schema(openapi, prefix, group, template)

    # ----------------- create prompts paths -----------------
    for name, prompt in prompts.items():
        await register_prompt_schema(openapi, prefix, group, prompt)

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
