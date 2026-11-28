package main

import (
	"bufio"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
)

type span struct {
	Kind  string `json:"kind"`
	Name  string `json:"name"`
	Text  string `json:"text"`
	Start int    `json:"start"`
	End   int    `json:"end"`
}

func index(source string) []span {
	fset := token.NewFileSet()
	tree, err := parser.ParseFile(fset, "input.go", source, 0)
	result := []span{}
	if err != nil {
		return nil
	}
	add := func(kind, name string, start, end token.Pos) {
		a, b := fset.Position(start), fset.Position(end)
		if a.IsValid() && b.IsValid() && a.Offset <= b.Offset && b.Offset <= len(source) {
			result = append(result, span{kind, name, strings.Join(strings.Fields(source[a.Offset:b.Offset]), " "), a.Line, b.Line})
		}
	}
	ast.Inspect(tree, func(node ast.Node) bool {
		switch n := node.(type) {
		case *ast.FuncDecl:
			end := n.End()
			if n.Body != nil {
				end = n.Body.Lbrace
			}
			name := n.Name.Name
			if n.Recv != nil {
				a, b := fset.Position(n.Recv.List[0].Type.Pos()), fset.Position(n.Recv.List[0].Type.End())
				name = source[a.Offset:b.Offset] + "." + name
			}
			kind := "function"
			if strings.HasPrefix(n.Name.Name, "Test") {
				kind = "test"
			}
			add(kind, name, n.Pos(), end)
			result = append(result, span{"scope", name, "", fset.Position(n.Pos()).Line, fset.Position(n.End()).Line})
		case *ast.IfStmt:
			add("condition", "", n.Cond.Pos(), n.Cond.End())
		case *ast.ForStmt:
			if n.Cond != nil {
				add("condition", "", n.Cond.Pos(), n.Cond.End())
			}
		case *ast.ImportSpec:
			add("import", n.Path.Value, n.Pos(), n.End())
		case *ast.GenDecl:
			if n.Tok == token.CONST {
				for _, spec := range n.Specs {
					if value, ok := spec.(*ast.ValueSpec); ok {
						for _, name := range value.Names {
							add("constant", name.Name, value.Pos(), value.End())
						}
					}
				}
			}
		}
		return true
	})
	return result
}

func main() {
	input := bufio.NewScanner(os.Stdin)
	input.Buffer(make([]byte, 65536), 16<<20)
	out := json.NewEncoder(os.Stdout)
	for input.Scan() {
		var source string
		if json.Unmarshal(input.Bytes(), &source) != nil {
			return
		}
		if out.Encode(index(source)) != nil {
			return
		}
	}
}
