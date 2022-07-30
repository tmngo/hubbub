#[cfg(test)]
mod tests {
    use crate::analyze::Analyzer;
    use crate::parse::Parser;
    use crate::tests::input::*;
    use crate::tokenize::Tokenizer;
    use crate::translate::cranelift::Generator;
    use crate::translate::input::Input;
    use crate::typecheck::Typechecker;
    use std::path::Path;

    pub fn test_jit(
        src: &str,
        expected_tree: &str,
        expected_definitions: usize,
        return_value: i64,
    ) {
        let source = src.to_string();

        let mut tokenizer = Tokenizer::new(&source);
        let tokens = tokenizer.tokenize();

        let mut parser = Parser::new(&source, tokens);
        parser.parse().ok();
        let tree = parser.tree();

        let result = format!("{}", tree);
        println!("{}", &result);
        if expected_tree != "" {
            assert_eq!(&result, expected_tree);
        }

        let mut analyzer = Analyzer::new(&tree);
        analyzer.resolve().ok();
        let definitions = analyzer.definitions;
        if expected_definitions != 0 {
            assert_eq!(definitions.len(), expected_definitions);
        }

        let mut typechecker = Typechecker::new(&tree, &definitions);
        typechecker.check().ok();
        let types = typechecker.types;
        let node_types = typechecker.node_types;

        let input = Input {
            tree: &tree,
            definitions: &definitions,
            types: &types,
            node_types: &node_types,
        };

        let generator = Generator::new(&input, "".to_string(), true);
        let result = generator.compile_nodes(Path::new(""));
        assert!(
            result.contains(&return_value),
            "expected main to return {:?}, got {:?}",
            return_value,
            result.unwrap()
        );
    }

    #[test]
    fn jit_struct() {
        test_jit(STRUCTS, STRUCTS_TREE, 23, 7);
    }

    #[test]
    fn jit_fibonacci() {
        test_jit(FIBONACCI, FIBONACCI_TREE, 12, 13);
    }

    #[test]
    fn jit_loop() {
        let source = &std::fs::read_to_string("../examples/loop.hb").unwrap();
        test_jit(source, "", 0, 10);
    }

    #[test]
    fn jit_main() {
        let source = &std::fs::read_to_string("../examples/main.hb").unwrap();
        test_jit(source, "", 0, 61);
    }

    #[test]
    fn jit_recursion() {
        let source = &std::fs::read_to_string("../examples/recursion.hb").unwrap();
        test_jit(source, "", 0, 13);
    }
}
