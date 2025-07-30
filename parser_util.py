from tree_sitter import Language, Parser
import tree_sitter_python

PY_LANGUAGE = Language(tree_sitter_python.language())# load the language

parser = Parser(PY_LANGUAGE)# create the parser


def parse_code(code: str):
    try:
        tree = parser.parse(bytes(code, 'utf8'))#parse the code and return the tree
        root = tree.root_node#get the root node
        if root.has_error or root.type == 'ERROR':#che ck the error
            return None
        return root#return the root node
    except Exception:
        return None
