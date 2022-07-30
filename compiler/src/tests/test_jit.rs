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
        assert_eq!(&result, expected_tree);

        let mut analyzer = Analyzer::new(&tree);
        analyzer.resolve().ok();
        let definitions = analyzer.definitions;
        assert_eq!(definitions.len(), expected_definitions);

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
        assert!(result.contains(&return_value));
    }

    #[test]
    fn jit_struct() {
        test_jit(STRUCTS, STRUCTS_TREE, 23, 7);
    }

    #[test]
    fn jit_fibonacci() {
        test_jit(FIBONACCI, FIBONACCI_TREE, 12, 13);
    }
}
