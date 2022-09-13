use std::path::Path;
use std::process::Command;

use crate::{
    analyze::Analyzer,
    link,
    parse::Parser,
    tests::input::*,
    tokenize::Tokenizer,
    translate::{cranelift::Generator, input::Input},
    typecheck::Typechecker,
};

pub enum Test {
    AotAndJit,
    Jit,
}

pub fn test(
    filename: &str,
    test: Test,
    expected_tree: &str,
    expected_definitions: usize,
    expected_exit_code: i64,
) {
    let path = Path::new("../examples/tests/")
        .join(filename)
        .with_extension("hb");
    let source = std::fs::read_to_string(path).unwrap();

    let mut tokenizer = Tokenizer::new(&source);
    let tokens = tokenizer.tokenize();
    let mut parser = Parser::new(&source, tokens);
    parser.parse().expect("Parse error");
    let tree = parser.tree();

    if expected_tree != "" {
        assert_eq!(expected_tree, &format!("{}", tree));
    }

    let mut analyzer = Analyzer::new(&tree);
    analyzer.resolve().expect("Name resolution error");
    let mut definitions = analyzer.definitions;

    if expected_definitions != 0 {
        assert_eq!(
            expected_definitions,
            definitions.len(),
            "expected {} definitions, got {}",
            expected_definitions,
            definitions.len()
        );
    }

    let mut typechecker = Typechecker::new(&tree, &mut definitions);
    typechecker.check().expect("Type error");
    let (types, node_types, type_parameters) = typechecker.results();
    let input = Input::new(&tree, &definitions, &types, &node_types, type_parameters);

    test_backend(filename, &input, true, expected_exit_code);
    if matches!(test, Test::AotAndJit) {
        test_backend(filename, &input, false, expected_exit_code);
    }
}

pub fn test_backend(filename: &str, input: &Input, use_jit: bool, expected_exit_code: i64) {
    let generator = Generator::new(&input, "".to_string(), use_jit);
    let obj_filename = format!("../test-{}.obj", filename);
    let obj_path = Path::new(&obj_filename);
    let result = generator.compile_nodes(obj_path);
    if use_jit {
        assert!(
            result.contains(&expected_exit_code),
            "expected main() to return {:?}, got {:?}",
            expected_exit_code,
            result.expect("main() returned nothing")
        );
    } else {
        let exe_filename = format!("../test-{}.exe", filename);
        let exe_path = Path::new(&exe_filename);
        link::link(&obj_filename, &exe_filename, "../");
        let output = Command::new(&exe_filename)
            .output()
            .expect("failed to execute");
        let exit_code = output.status.code();
        assert!(
            exit_code.contains(&(expected_exit_code as i32)),
            "expected main() to return {:?}, got {:?}",
            expected_exit_code,
            exit_code.expect("main() returned nothing")
        );
        if obj_path.exists() {
            std::fs::remove_file(obj_path).ok();
        }
        if exe_path.exists() {
            std::fs::remove_file(exe_path).ok();
        }
    }
}

#[test]
fn boolean() {
    test("boolean", Test::Jit, "", 0, 6);
}

#[test]
fn structs() {
    test("struct", Test::Jit, STRUCTS_TREE, 23, 7);
}

#[test]
fn while_loop() {
    test("loop", Test::Jit, "", 0, 35);
}

#[test]
fn main() {
    test("main", Test::Jit, "", 0, 61);
}

#[test]
fn fibonacci() {
    test("fibonacci", Test::Jit, FIBONACCI_TREE, 11, 13)
}

#[test]
fn unicode() {
    test("unicode", Test::AotAndJit, "", 0, 114);
}
