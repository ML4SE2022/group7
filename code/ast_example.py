import ast

# code = "def writeBoolean(self, n):\n        \"\"\"\n        Writes a Boolean to the stream.\n        \"\"\"\n        t = TYPE_BOOL_TRUE\n\n        if n is False:\n            t = TYPE_BOOL_FALSE\n\n        self.stream.write(t)"
# code = "def split_phylogeny ( p , level = \"s\" ) :\n level = level + \"__\"\n result = p . split ( level )\n return result [ 0 ] + level + result [ 1 ] . split ( \";\" ) [ 0 ]"
code = "def ensure_dir ( d ) :\n if not os . path . exists ( d ) :\n \r try : os . makedirs ( d ) except OSError as oe : # should not happen with os.makedirs # ENOENT: No such file or directory if os . errno == errno . ENOENT : msg = twdd ( \"\"\"One or more directories in the path ({}) do not exist. If                            you are specifying a new directory for output, please ensure                            all other directories in the path currently exist.\"\"\" ) return msg . format ( d ) else : msg = twdd ( \"\"\"An error occurred trying to create the output directory                            ({}) with message: {}\"\"\" ) return msg . format ( d , oe . strerror )"
code_ast = ast.parse(code)
dict = []
res = []
for k in ast.walk(code_ast):
    print(k.__class__.__name__)
    if dict.count(k.__class__.__name__) == 0:
        dict.append(k.__class__.__name__)
    res.append(dict.index(k.__class__.__name__))
print(res)
print(ast.dump(code_ast))
