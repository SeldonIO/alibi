#!/usr/bin/env python3
"""
Generate GitBook-ready Markdown API docs for the `alibi` repo (or any Python package
with subpackages, classes, and methods).

Key features:
- Recursively walks a package (default: `alibi`) without hardcoding targets.
- Respects a module's `__all__` when present to define the public API.
- Renders modules, classes, dataclasses, properties, methods, and functions.
- Parses docstrings (prefers `docstring_parser` if installed; falls back to simple parsing).
- Includes parameter tables, return types, and return descriptions when available.
- Emits a GitBook `SUMMARY.md` and one Markdown file per module under `api/`.
- Optional: include inherited members, private members, or exclude modules by pattern.
- Optional: "View source" links pointing at your repo host.
- Optional: prepend sys.path entries so you can run against a local checkout without installing.

Usage (from repo root or any environment where `alibi` is importable):
    python generate_alibi_api_docs.py --package alibi --outdir docs-gb
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import importlib
import inspect
import os
import pkgutil
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union, get_origin, get_args

# ---------------------------
# Optional docstring parsing
# ---------------------------
_DOCSTRING_PARSER = None
try:
    import docstring_parser as _docstring_parser  # type: ignore
    _DOCSTRING_PARSER = _docstring_parser
except Exception:
    _DOCSTRING_PARSER = None

# ---------------------------
# Utilities
# ---------------------------
def debug(msg: str, verbose: bool):
    if verbose:
        print(f"[generate-alibi-docs] {msg}")

def is_public_name(name: str) -> bool:
    return not name.startswith("_")

def is_same_module(obj: Any, module_name: str) -> bool:
    return getattr(obj, "__module__", None) == module_name

def safe_get_module_all(mod: ModuleType) -> Optional[List[str]]:
    try:
        all_ = getattr(mod, "__all__", None)
        if all_ and isinstance(all_, (list, tuple)):
            return list(all_)
    except Exception:
        pass
    return None

def type_to_str(tp: Any) -> str:
    """Return a readable string for a type annotation, including typing constructs."""
    if tp is None:
        return "None"
    if tp is inspect._empty:
        return ""
    # Handle strings (ForwardRef or stringified annotation)
    if isinstance(tp, str):
        return tp

    origin = get_origin(tp)
    args = get_args(tp)

    # Bare types or typing.Any / builtins
    if origin is None:
        mod = getattr(tp, "__module__", "")
        name = getattr(tp, "__qualname__", getattr(tp, "__name__", str(tp)))
        if mod in ("builtins", "typing"):
            return name.replace("NoneType", "None")
        return f"{mod}.{name}".replace("NoneType", "None")

    # typing constructs
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(args) == 2 and len(non_none) == 1:
            return f"Optional[{type_to_str(non_none[0])}]"
        return "Union[" + ", ".join(type_to_str(a) for a in args) + "]"

    name = getattr(origin, "_name", None) or getattr(origin, "__name__", str(origin))
    if name in ("List", "list"):
        return f"List[{type_to_str(args[0])}]" if args else "List[Any]"
    if name in ("Tuple", "tuple"):
        return "Tuple[" + ", ".join(type_to_str(a) for a in args) + "]" if args else "Tuple"
    if name in ("Dict", "dict"):
        if len(args) == 2:
            return f"Dict[{type_to_str(args[0])}, {type_to_str(args[1])}]"
        return "Dict"
    if name in ("Callable", "collections.abc.Callable"):
        if args:
            *params, ret = args
            if len(params) == 1 and params[0] is Ellipsis:
                params_str = "..."
            else:
                params_str = ", ".join(type_to_str(p) for p in params)
            return f"Callable[[{params_str}], {type_to_str(ret)}]"
        return "Callable"
    # Generic fallback
    if args:
        return f"{name}[" + ", ".join(type_to_str(a) for a in args) + "]"
    return name

def format_signature(func: Any) -> str:
    """Render a signature with annotations in a compressed style suitable for GitBook."""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        name = getattr(func, "__name__", "<callable>")
        return f"{name}(...)"
    params_out = []
    hints = {}
    try:
        hints = typing_get_type_hints_safe(func)
    except Exception:
        pass
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        ann = hints.get(name, param.annotation)
        ann_str = f": {type_to_str(ann)}" if ann is not inspect._empty else ""
        if param.default is not inspect._empty:
            params_out.append(f"{name}{ann_str} = {repr(param.default)}")
        else:
            params_out.append(f"{name}{ann_str}")
    ret_ann = hints.get("return", sig.return_annotation)
    ret_str = f" -> {type_to_str(ret_ann)}" if ret_ann is not inspect._empty else ""
    return f"{getattr(func, '__name__', '<callable>')}(" + ", ".join(params_out) + ")" + ret_str

def typing_get_type_hints_safe(obj: Any) -> Dict[str, Any]:
    """Get type hints but avoid crashing on unresolved forward refs."""
    try:
        import typing
        globalns = {}
        localns = {}
        mod = inspect.getmodule(obj)
        if mod is not None:
            globalns = dict(getattr(mod, "__dict__", {}))
        if inspect.ismethod(obj) or (inspect.isfunction(obj) and "." in obj.__qualname__):
            cls_name = obj.__qualname__.split(".")[0]
            if cls_name and cls_name in globalns:
                localns = dict(getattr(globalns[cls_name], "__dict__", {}))
        return typing.get_type_hints(obj, globalns=globalns, localns=localns)
    except Exception:
        return {}

def parse_docstring(doc: Optional[str]) -> Dict[str, Any]:
    """Parse docstring into a structured dict. Uses docstring_parser if available."""
    result = {
        "short": "",
        "long": "",
        "params": [],
        "returns": None,
        "raises": [],
        "examples": []
    }
    if not doc:
        return result

    if _DOCSTRING_PARSER is not None:
        try:
            parsed = _DOCSTRING_PARSER.parse(doc, style=_DOCSTRING_PARSER.DocstringStyle.NUMPYDOC)
            
            result["short"] = (parsed.short_description or "").strip()
            result["long"] = (parsed.long_description or "").strip()

            for p in parsed.params:
                result["params"].append({
                    "name": p.arg_name or "",
                    "type": (p.type_name or "").strip(),
                    "default": (p.default or "").strip(),
                    "desc": (p.description or "").strip(),
                })

            if parsed.returns:
                result["returns"] = {
                    "type": (parsed.returns.type_name or "").strip(),
                    "desc": (parsed.returns.description or "").strip(),
                }

            for r in parsed.raises:
                result["raises"].append({
                    "type": (r.type_name or "").strip(),
                    "desc": (r.description or "").strip(),
                })

            for meta in getattr(parsed, "meta", []):
                if str(meta.args or [""])[0].lower().startswith("example"):
                    if meta.description:
                        result["examples"].append(meta.description.strip())
            
            if result["params"] and any(not p.get("desc") for p in result["params"]):
                param_match = re.search(r'Parameters\s*\n\s*-+\s*\n(.*?)(?=\n\s*(?:Returns?|Raises?|Yields?|Examples?|Notes?|See Also)\s*\n\s*-+|$)', 
                                       doc, re.DOTALL | re.IGNORECASE)
                if param_match:
                    param_section = param_match.group(1)
                    param_map = {p["name"]: p for p in result["params"]}
                    
                    lines = param_section.split('\n')
                    current_param = None
                    desc_lines = []
                    
                    for line in lines:
                        if line and not line[0].isspace():
                            if current_param and current_param in param_map:
                                param_map[current_param]["desc"] = ' '.join(desc_lines).strip()
                            
                            param_name = line.strip().split()[0] if line.strip() else None
                            if param_name and param_name in param_map:
                                current_param = param_name
                                desc_lines = []
                            else:
                                current_param = None
                        elif line.strip() and current_param:
                            desc_lines.append(line.strip())
                    
                    if current_param and current_param in param_map:
                        param_map[current_param]["desc"] = ' '.join(desc_lines).strip()
            
            return result

        except Exception:
            pass

    lines = doc.strip().splitlines()
    result["short"] = lines[0].strip() if lines else ""
    if len(lines) > 1:
        result["long"] = "\n".join(lines[1:]).strip()

    return result

def render_params_table(params: List[Dict[str, str]], sig: Optional[inspect.Signature], hints: Dict[str, Any]) -> str:
    """Render a Markdown table of parameters. Merge docstring info with signature types/defaults."""
    if not params and not sig:
        return ""
    ds_map: Dict[str, Dict[str, str]] = {p["name"]: p for p in params if p.get("name")}
    rows: List[Tuple[str, str, str, str]] = []
    seen = set()
    if sig:
        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                ds = ds_map.get(name, {})
                if not ds.get("type") and not ds.get("default") and not ds.get("desc"):
                    continue
            ds = ds_map.get(name, {})
            ann = hints.get(name, param.annotation)
            typ = ds.get("type") or (type_to_str(ann) if ann is not inspect._empty else "")
            default = ds.get("default") or (repr(param.default) if param.default is not inspect._empty else "")
            desc = ds.get("desc", "")
            rows.append((name, typ, default, desc))
            seen.add(name)
    for name, ds in ds_map.items():
        if name in seen:
            continue
        rows.append((name, ds.get("type", ""), ds.get("default", ""), ds.get("desc", "")))

    if not rows:
        return ""

    out = []
    out.append("| Name | Type | Default | Description |")
    out.append("| ---- | ---- | ------- | ----------- |")
    for name, typ, default, desc in rows:
        name_md = f"`{name}`"
        typ_md = f"`{typ}`" if typ else ""
        default_md = f"`{default}`" if default else ""
        desc_md = desc.replace("\n", " ").strip()
        out.append(f"| {name_md} | {typ_md} | {default_md} | {desc_md} |")
    return "\n".join(out)

def render_returns_block(returns: Optional[Dict[str, str]], sig: Optional[inspect.Signature], hints: Dict[str, Any]) -> str:
    typ = ""
    desc = ""
    if returns:
        typ = returns.get("type", "") or ""
        desc = (returns.get("desc", "") or "").strip()
    if not typ and sig:
        ann = hints.get("return", sig.return_annotation)
        if ann is not inspect._empty:
            typ = type_to_str(ann)
    if not typ and not desc:
        return ""
    out = ["**Returns**"]
    if typ:
        out.append(f"- Type: `{typ}`")
    if desc:
        out.append(f"- {desc}")
    return "\n".join(out)

def get_properties(cls: type) -> List[Tuple[str, Any]]:
    props: List[Tuple[str, Any]] = []
    for name, obj in cls.__dict__.items():
        if isinstance(obj, property) and is_public_name(name):
            props.append((name, obj))
    return props

def get_methods(cls: type, include_inherited: bool) -> List[Tuple[str, Any]]:
    methods = []
    for name, obj in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not is_public_name(name):
            continue
        if not include_inherited and obj.__qualname__.split(".")[0] != cls.__name__:
            continue
        methods.append((name, obj))
    for name, obj in cls.__dict__.items():
        if not is_public_name(name):
            continue
        if isinstance(obj, (staticmethod, classmethod)):
            fn = obj.__func__
            if (name, fn) not in methods:
                methods.append((name, fn))
    methods.sort(key=lambda x: (x[0] == "__call__", x[0]))
    return methods

def make_source_link(obj: Any, repo_root: Optional[str], source_url_prefix: Optional[str]) -> Optional[str]:
    if not source_url_prefix or not obj:
        return None
    try:
        import inspect
        file = inspect.getsourcefile(obj) or inspect.getfile(obj)
        lines, start = inspect.getsourcelines(obj)
        end = start + len(lines) - 1
        file_path = Path(file).resolve()
        if repo_root:
            try:
                rel = file_path.relative_to(Path(repo_root).resolve())
            except Exception:
                rel = file_path.name
        else:
            rel = file_path.name
        return f"{source_url_prefix.rstrip('/')}/{rel.as_posix()}#L{start}-L{end}"
    except Exception:
        return None

def render_module(mod: ModuleType, include_inherited: bool, verbose: bool, repo_root: Optional[str], source_url_prefix: Optional[str]) -> str:
    parts = []
    title = f"# `{mod.__name__}`"
    parts.append(title)
    mod_ds = parse_docstring(inspect.getdoc(mod))
    if mod_ds["short"] or mod_ds["long"]:
        parts.append("")
        if mod_ds["short"]:
            parts.append(mod_ds["short"])
        if mod_ds["long"]:
            parts.append(mod_ds["long"])
        parts.append("")
    
    classes, funcs = select_public_members(mod, want_classes=True, want_funcs=True)
    if classes:
        for name, cls in classes:
            parts.append(render_class(cls, include_inherited=include_inherited, verbose=verbose, repo_root=repo_root, source_url_prefix=source_url_prefix))
    if funcs:
        parts.append("## Functions")
        for name, fn in funcs:
            parts.append(render_function(name, fn, repo_root=repo_root, source_url_prefix=source_url_prefix))
    return "\n".join(parts).strip() + "\n"

def render_class(cls: type, include_inherited: bool, verbose: bool, repo_root: Optional[str] = None, source_url_prefix: Optional[str] = None) -> str:
    out = []
    out.append(f"## `{cls.__name__}`\n")
    base_names = [b.__name__ for b in getattr(cls, "__mro__", [])[1:] if b not in (object,)]
    if base_names:
        out.append(f"_Inherits from:_ {', '.join('`' + b + '`' for b in base_names)}\n")
    link = make_source_link(cls, repo_root, source_url_prefix)
    if link:
        out.append(f"[View source]({link})\n")
    class_doc = inspect.getdoc(cls)
    is_inherited_doc = False
    if class_doc:
        for base in cls.__mro__[1:]:
            if base is object:
                continue
            base_doc = inspect.getdoc(base)
            if base_doc and base_doc == class_doc:
                is_inherited_doc = True
                break
    if not is_inherited_doc:
        class_ds = parse_docstring(class_doc)
        if class_ds["short"]:
            out.append(class_ds["short"] + "\n")
        if class_ds["long"]:
            out.append(class_ds["long"] + "\n")
    if dataclasses.is_dataclass(cls):
        fields = dataclasses.fields(cls)
        if fields:
            out.append("### Fields\n")
            out.append("| Field | Type | Default |")
            out.append("| ----- | ---- | ------- |")
            for f in fields:
                typ = type_to_str(f.type) if f.type is not dataclasses.MISSING else ""
                default = ""
                if f.default is not dataclasses.MISSING:
                    default = repr(f.default)
                elif f.default_factory is not dataclasses.MISSING:
                    default = f"{f.default_factory}()"
                out.append(f"| `{f.name}` | `{typ}` | `{default}` |")
            out.append("")
    init = getattr(cls, "__init__", None)
    if callable(init):
        is_inherited_init = False
        for base in cls.__mro__[1:]:
            if base is object:
                continue
            base_init = getattr(base, "__init__", None)
            if base_init and base_init is init:
                is_inherited_init = True
                break
        if not is_inherited_init:
            sig = None
            hints = {}
            try:
                sig = inspect.signature(init)
                hints = typing_get_type_hints_safe(init)
            except Exception:
                pass
            out.append("### Constructor\n")
            if sig:
                out.append(f"```python\n{cls.__name__}{sig}\n```")
            else:
                out.append(f"```python\n{cls.__name__}(...)\n```")
            init_ds = parse_docstring(inspect.getdoc(init))
            params_table = render_params_table(init_ds["params"], sig, hints)
            if params_table:
                out.append("\n" + params_table + "\n")
    props = get_properties(cls)
    if props:
        out.append("### Properties\n")
        out.append("| Property | Type | Description |")
        out.append("| -------- | ---- | ----------- |")
        for prop_name, prop in props:
            ann = getattr(prop.fget, "__annotations__", {}).get("return", "")
            typ = type_to_str(ann) if ann else ""
            pdoc = parse_docstring(inspect.getdoc(prop.fget))
            desc = pdoc["short"] or pdoc["long"]
            out.append(f"| `{prop_name}` | `{typ}` | {desc} |")
        out.append("")
    methods = get_methods(cls, include_inherited=include_inherited)
    if methods:
        out.append("### Methods\n")
        for name, fn in methods:
            if name.startswith("_") and name != "__call__":
                continue
            fn_doc = inspect.getdoc(fn)
            parent_abstract_doc = None
            for base in cls.__mro__[1:]:
                if base is object:
                    continue
                base_method = getattr(base, name, None)
                if base_method and callable(base_method):
                    if hasattr(base_method, '__isabstractmethod__') and base_method.__isabstractmethod__:
                        parent_abstract_doc = inspect.getdoc(base_method)
                        break
            use_doc = fn_doc
            if not fn_doc and parent_abstract_doc:
                use_doc = parent_abstract_doc
            is_inherited_method_doc = False
            if use_doc and not parent_abstract_doc:
                for base in cls.__mro__[1:]:
                    if base is object:
                        continue
                    base_method = getattr(base, name, None)
                    if base_method and callable(base_method):
                        base_doc = inspect.getdoc(base_method)
                        if base_doc and base_doc == use_doc:
                            if not (hasattr(base_method, '__isabstractmethod__') and base_method.__isabstractmethod__):
                                is_inherited_method_doc = True
                                break
            fn_ds = parse_docstring(use_doc if not is_inherited_method_doc else None)
            sig = None
            hints = {}
            try:
                sig = inspect.signature(fn)
                hints = typing_get_type_hints_safe(fn)
            except Exception:
                pass
            sig_str = format_signature(fn)
            out.append(f"#### `{name}`\n")
            out.append(f"```python\n{sig_str}\n```\n")
            link = make_source_link(fn, repo_root, source_url_prefix)
            if link:
                out.append(f"[View source]({link})\n")
            if not is_inherited_method_doc:
                if fn_ds["short"]:
                    out.append(fn_ds["short"] + "\n")
                if fn_ds["long"]:
                    out.append(fn_ds["long"] + "\n")
            params_table = render_params_table(fn_ds["params"], sig, hints)
            if params_table:
                out.append(params_table + "\n")
            ret_block = render_returns_block(fn_ds["returns"], sig, hints)
            if ret_block:
                out.append(ret_block + "\n")
            if fn_ds["raises"]:
                out.append("**Raises**")
                for r in fn_ds["raises"]:
                    typ = f"`{r['type']}`" if r.get("type") else ""
                    desc = r.get("desc", "")
                    out.append(f"- {typ} {desc}".strip())
                out.append("")
            if fn_ds["examples"]:
                out.append("**Examples**")
                for ex in fn_ds["examples"]:
                    out.append("```python")
                    out.append(ex.strip())
                    out.append("```")
                out.append("")
    return "\n".join(out).strip() + "\n"

def render_function(name: str, fn: Any, repo_root: Optional[str] = None, source_url_prefix: Optional[str] = None) -> str:
    out = []
    ds = parse_docstring(inspect.getdoc(fn))
    sig_str = format_signature(fn)
    out.append(f"### `{name}`\n")
    out.append(f"```python\n{sig_str}\n```\n")
    link = make_source_link(fn, repo_root, source_url_prefix)
    if link:
        out.append(f"[View source]({link})\n")
    if ds["short"]:
        out.append(ds["short"] + "\n")
    if ds["long"]:
        out.append(ds["long"] + "\n")
    sig = None
    hints = {}
    try:
        sig = inspect.signature(fn)
        hints = typing_get_type_hints_safe(fn)
    except Exception:
        pass
    params_table = render_params_table(ds["params"], sig, hints)
    if params_table:
        out.append(params_table + "\n")
    ret_block = render_returns_block(ds["returns"], sig, hints)
    if ret_block:
        out.append(ret_block + "\n")
    if ds["raises"]:
        out.append("**Raises**")
        for r in ds["raises"]:
            typ = f"`{r['type']}`" if r.get("type") else ""
            desc = r.get("desc", "")
            out.append(f"- {typ} {desc}".strip())
        out.append("")
    if ds["examples"]:
        out.append("**Examples**")
        for ex in ds["examples"]:
            out.append("```python")
            out.append(ex.strip())
            out.append("```")
        out.append("")
    return "\n".join(out).strip() + "\n"

def should_skip_module(mod_name: str, include_private: bool, exclude_globs: List[str]) -> bool:
    if not include_private and any(part.startswith("_") for part in mod_name.split(".")):
        return True
    for pat in exclude_globs:
        if fnmatch.fnmatch(mod_name, pat):
            return True
    if ".tests" in mod_name or mod_name.endswith(".tests"):
        return True
    return False

def walk_package(package: str, verbose: bool) -> List[str]:
    try:
        pkg = importlib.import_module(package)
    except Exception as e:
        raise SystemExit(f"Could not import package '{package}': {e}")
    paths = getattr(pkg, "__path__", None)
    if paths is None:
        return [package]
    mods = []
    for finder, name, ispkg in pkgutil.walk_packages(paths, prefix=pkg.__name__ + "."):
        mods.append(name)
    return sorted([package] + mods)

def import_module_safely(mod_name: str, verbose: bool) -> Optional[ModuleType]:
    try:
        return importlib.import_module(mod_name)
    except Exception as e:
        print(f"[warn] Skipping module '{mod_name}' due to import error: {e}")
        return None

def select_public_members(mod: ModuleType, want_classes: bool = True, want_funcs: bool = True) -> Tuple[List[Tuple[str, Any]], List[Tuple[str, Any]]]:
    classes: List[Tuple[str, Any]] = []
    funcs: List[Tuple[str, Any]] = []
    allow = safe_get_module_all(mod)
    members = inspect.getmembers(mod)
    for name, obj in members:
        if allow is not None and name not in allow:
            continue
        if allow is None and not is_public_name(name):
            continue
        if want_classes and inspect.isclass(obj) and is_same_module(obj, mod.__name__):
            classes.append((name, obj))
        if want_funcs and inspect.isfunction(obj) and is_same_module(obj, mod.__name__):
            funcs.append((name, obj))
    classes.sort(key=lambda x: x[0])
    funcs.sort(key=lambda x: x[0])
    return classes, funcs

def write_api_summary(all_module_names: List[str], outdir: Path, filename: str = "SUMMARY-API.md"):
    lines: List[str] = []
    lines.append("* API Reference")
    for modname in sorted(all_module_names):
        depth = modname.count(".") + 1
        rel_md = f"api/{modname.replace('.', '/')}.md"
        lines.append("  " * depth + f"* [`{modname}`]({rel_md})")
    (outdir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Generate GitBook API docs from Python package.")
    parser.add_argument("--package", default="alibi", help="Package name to document")
    parser.add_argument("--outdir", default="docs-gb", help="Output directory")
    parser.add_argument("--include-inherited", action="store_true", help="Include inherited members")
    parser.add_argument("--include-private", action="store_true", help="Include private modules")
    parser.add_argument("--exclude", nargs="*", default=[], help="Glob patterns to exclude modules")
    parser.add_argument("--repo-root", help="Local repo root for computing source links")
    parser.add_argument("--source-url-prefix", help="URL prefix for source links, e.g., https://github.com/SeldonIO/alibi/blob/main")
    parser.add_argument("--prepend-path", nargs="*", default=[], help="Paths to prepend to sys.path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    for p in args.prepend_path:
        sys.path.insert(0, str(Path(p).resolve()))

    outdir = Path(args.outdir)
    api_dir = outdir / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

    mods = walk_package(args.package, args.verbose)
    written = []
    for mod_name in mods:
        if should_skip_module(mod_name, args.include_private, args.exclude):
            debug(f"Skipping {mod_name}", args.verbose)
            continue
        mod = import_module_safely(mod_name, args.verbose)
        if mod is None:
            continue
        debug(f"Rendering {mod_name}", args.verbose)
        content = render_module(mod, args.include_inherited, args.verbose, args.repo_root, args.source_url_prefix)
        out_file = api_dir / f"{mod_name.replace('.', '/')}.md"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(content, encoding="utf-8")
        written.append(mod_name)

    write_api_summary(written, outdir)
    print(f"Generated {len(written)} module docs in {api_dir}")
    print(f"API summary written to {outdir / 'SUMMARY-API.md'}")

if __name__ == "__main__":
    main()