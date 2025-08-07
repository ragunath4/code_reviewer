from tree_sitter import Language, Parser
import tree_sitter_python

PY_LANGUAGE = Language(tree_sitter_python.language())  # load the language

parser = Parser(PY_LANGUAGE)  # create the parser


def has_tree_error(node):
    if node.has_error or node.type == "ERROR":
        return True
    for child in node.children:
        if has_tree_error(child):
            return True
    return False


def parse_code(code: str):
    try:
        # parse the code and return the tree
        tree = parser.parse(bytes(code, 'utf8'))
        root = tree.root_node  # get the root node
        # Always return the root, even if it has errors
        return root  # return the root node
    except Exception:
        return None
