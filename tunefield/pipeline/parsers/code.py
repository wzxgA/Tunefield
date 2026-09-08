"""T2 · 源代码解析器（tree-sitter）：函数/类/接口等定义边界 → 源码块。

设计（对齐 B2.2「源代码」形态）：
- 语言由扩展名识别；只取「定义节点」整体（先序、命中即不再下钻），
  一个函数 / 类 / 方法 / 接口即一块，保证 T4 切片不把定义切半；
- 每块携带 language / 路径 / 定义类型 / 符号名 / 起止行元信息；
- tree-sitter 语法包惰性加载：某语言缺失或解析异常时，回退为「整文件单块」
  （结构边界丢失，内容仍可预览、可入语料），不阻断管线。

依赖版本说明：tree-sitter 核心与各语法包需配对使用（0.25.x 核心 + 各语言包），
核心升到 0.26 后 capsule API 不兼容，见 pyproject 中的版本约束。
"""

from __future__ import annotations

import importlib
import re

from .txt import _decode

# 扩展名 → 语言名（语言名是树节点类型集合与语法包 loader 的键）
_LANG_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".rs": "rust",
}

# 语言 → (语法包模块, 包内 language 导出函数名)
_LANG_LOADER: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "java": ("tree_sitter_java", "language"),
    "go": ("tree_sitter_go", "language"),
    "c": ("tree_sitter_cpp", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "rust": ("tree_sitter_rust", "language"),
}

# 各语言视为「定义边界」的节点类型
_TS_DEF_NODES = {
    "function_declaration",
    "class_declaration",
    "method_definition",
    "interface_declaration",
    "enum_declaration",
    "function_signature",
    "method_signature",
    "record_declaration",
    "constructor_declaration",
}
_DEF_NODES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": _TS_DEF_NODES,
    "tsx": _TS_DEF_NODES,
    "java": {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
        "method_declaration",
        "constructor_declaration",
    },
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "c": {"function_definition", "struct_specifier", "enum_specifier"},
    "cpp": {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "enum_specifier",
    },
    "rust": {
        "function_item",
        "struct_item",
        "enum_item",
        "impl_item",
        "trait_item",
        "mod_item",
    },
}

_NAME_FIELD_TYPES = {"identifier", "type_identifier", "property_identifier"}


def language_for(ext: str) -> str | None:
    """扩展名 → 语言名（未收录返回 None，归为不支持/附加文本）。"""
    return _LANG_BY_EXT.get(ext.lower())


def _collect_defs(root, def_types: set[str]) -> list:
    """先序收集定义节点：命中定义即整体收入、不再下钻（保证不切半）。"""
    out: list = []

    def walk(node) -> None:
        if node.type in def_types:
            out.append(node)
            return
        for child in node.children:
            walk(child)

    walk(root)
    return out


def _node_symbol(node, node_text: bytes) -> str:
    """尽力取符号名；取不到时回退为签名首行摘要（如 C 函数定义）。"""
    named = node.child_by_field_name("name")
    if named is not None and named.text:
        name = named.text.decode("utf-8", "replace").strip()
        if name:
            return name
    for child in node.children:
        if child.type in _NAME_FIELD_TYPES and child.text:
            name = child.text.decode("utf-8", "replace").strip()
            if name:
                return name
    first = node_text.splitlines()[0].decode("utf-8", "replace").strip()
    return first[:40] or "(anonymous)"


def parse(name: str, data: bytes, language: str) -> dict:
    text = _decode(data)
    if b"\x00" in data or not text.strip():
        return {
            "status": "empty",
            "error": "疑似二进制或空文件",
            "chars": 0,
            "segments": [],
        }

    base_meta: dict = {"kind": "source", "language": language, "path": name}
    structure = "text"  # 回退：整文件文本块

    defs: list = []
    try:
        mod_name, attr = _LANG_LOADER[language]
        grammar_mod = importlib.import_module(mod_name)
        load_fn = getattr(grammar_mod, attr)
        from tree_sitter import Language, Parser

        lang = Language(load_fn())
        tree = Parser(lang).parse(text.encode("utf-8"))
        defs = _collect_defs(tree.root_node, _DEF_NODES.get(language, set()))
        structure = "tree-sitter"
    except Exception:
        defs = []  # 语法缺失 / ABI 不匹配 / 解析异常 → 文本回退

    segments: list[dict] = []
    if not defs:
        segments.append({"index": 0, "text": text.strip(), "meta": {**base_meta}})
        return {
            "status": "ok",
            "chars": len(text),
            "structure": structure,
            "segments": segments,
        }

    for node in defs:
        node_bytes = node.text
        if not node_bytes:
            continue
        block_text = node_bytes.decode("utf-8", "replace").strip()
        if not block_text:
            continue
        segments.append(
            {
                "index": len(segments),
                "text": block_text,
                "meta": {
                    **base_meta,
                    "definition": node.type,
                    "symbol": _node_symbol(node, node_bytes),
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                },
            }
        )

    if not segments:
        segments.append({"index": 0, "text": text.strip(), "meta": {**base_meta}})
    return {"status": "ok", "chars": len(text), "structure": structure, "segments": segments}
