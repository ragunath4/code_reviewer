from tree_sitter import Language, Parser
import tree_sitter_python

PY_LANGUAGE = Language(tree_sitter_python.language())

parser = Parser(PY_LANGUAGE)


def parse_code(code: str):
    try:
        tree = parser.parse(bytes(code, 'utf8'))
        root = tree.root_node
        if root.has_error or root.type == 'ERROR':
            return None
        return root
    except Exception:
        return None
