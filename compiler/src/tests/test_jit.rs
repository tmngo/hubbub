use crate::analyze::Analyzer;
use crate::parse::Parser;
use crate::tests::input::*;
use crate::tokenize::Tokenizer;
use crate::translate::cranelift::Generator;
use crate::translate::input::Input;
use crate::typecheck::Typechecker;
use std::path::Path;

pub fn test_jit(src: &str, expected_tree: &str, expected_definitions: usize, return_value: i64) {
    let source = src.to_string();

    let mut tokenizer = Tokenizer::new(&source);
    let tokens = tokenizer.tokenize();

    let mut parser = Parser::new(&source, tokens);
    parser.parse().expect("Parse error");
    let tree = parser.tree();

    let result = format!("{}", tree);
    println!("{}", &result);
    if expected_tree != "" {
        assert_eq!(&result, expected_tree);
    }

    let mut analyzer = Analyzer::new(&tree);
    analyzer.resolve().expect("Name resolution error");
    let definitions = analyzer.definitions;
    if expected_definitions != 0 {
        assert_eq!(
            definitions.len(),
            expected_definitions,
            "expected {} definitions, got {}",
            expected_definitions,
            definitions.len()
        );
    }

    let mut typechecker = Typechecker::new(&tree, &definitions);
    typechecker.check().expect("Type error");
    let (types, node_types, type_parameters) = typechecker.results();

    let input = Input::new(&tree, &definitions, &types, &node_types, type_parameters);

    let generator = Generator::new(input, "".to_string(), true);
    let result = generator.compile_nodes(Path::new(""));
    assert!(
        result.contains(&return_value),
        "expected main() to return {:?}, got {:?}",
        return_value,
        result.expect("main() returned nothing")
    );
}

#[test]
fn jit_boolean() {
    let source = &std::fs::read_to_string("../examples/tests/boolean.hb").unwrap();
    test_jit(source, "", 0, 6);
}

#[test]
fn jit_struct() {
    let source = &std::fs::read_to_string("../examples/tests/struct.hb").unwrap();
    test_jit(source, STRUCTS_TREE, 23, 7);
}

#[test]
fn jit_loop() {
    let source = &std::fs::read_to_string("../examples/tests/loop.hb").unwrap();
    test_jit(source, "", 0, 35);
}

#[test]
fn jit_main() {
    let source = &std::fs::read_to_string("../examples/tests/main.hb").unwrap();
    test_jit(source, "", 0, 61);
}

#[test]
fn jit_fibonacci() {
    let source = &std::fs::read_to_string("../examples/tests/fibonacci.hb").unwrap();
    test_jit(source, FIBONACCI_TREE, 11, 13);
}
