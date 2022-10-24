use crate::{
    analyze::Analyzer,
    link::{link, prepend_module, set_default_absolute_module_path},
    parse::Parser,
    tests::input::*,
    tokenize::Tokenizer,
    translate::{cranelift::Generator, input::Input, llvm},
    typecheck::Typechecker,
    workspace::Workspace,
};
use std::{path::Path, process::Command};

pub enum Test {
    AotAndJit,
    // Jit,
    WithPrelude,
}

pub enum Backend {
    Cranelift,
    Llvm,
}

pub fn test(
    filename: &str,
    test: Test,
    expected_tree: &str,
    expected_definitions: usize,
    expected_exit_code: i64,
) {
    set_default_absolute_module_path();
    let path = Path::new("../examples/tests/")
        .join(filename)
        .with_extension("hb");
    let source = std::fs::read_to_string(path).unwrap();
    let source = if matches!(test, Test::WithPrelude) {
        prepend_module("Prelude.hb", &source)
    } else {
        source
    };

    let mut tokenizer = Tokenizer::new(&source);
    let tokens = tokenizer.tokenize();

    let mut workspace = Workspace::new();

    let mut parser = Parser::new(&mut workspace, &source, tokens);
    parser.parse();
    let tree = parser.tree();
    let formatted_tree = &format!("{:?}", tree);
    if !expected_tree.is_empty() {
        assert_eq!(
            expected_tree, formatted_tree,
            "expected:\n{}\n\ngot:\n{}\n",
            expected_tree, formatted_tree
        );
    }

    let mut analyzer = Analyzer::new(&mut workspace, &tree);
    analyzer.resolve().expect("Name resolution error");
    let mut definitions = analyzer.definitions;
    let overload_sets = analyzer.overload_sets;

    if expected_definitions != 0 {
        assert_eq!(
            expected_definitions,
            definitions.len(),
            "expected {} definitions, got {}",
            expected_definitions,
            definitions.len()
        );
    }

    let mut typechecker = Typechecker::new(&mut workspace, &tree, &mut definitions, &overload_sets);
    typechecker.check().expect("Type error");
    let (types, node_types, type_parameters) = typechecker.results();
    let input = Input::new(&tree, &definitions, &types, &node_types, type_parameters);

    test_backend(
        Backend::Cranelift,
        &workspace,
        filename,
        &input,
        true,
        expected_exit_code,
    );
    if matches!(test, Test::AotAndJit | Test::WithPrelude) {
        test_backend(
            Backend::Cranelift,
            &workspace,
            filename,
            &input,
            false,
            expected_exit_code,
        );
        test_backend(
            Backend::Llvm,
            &workspace,
            filename,
            &input,
            false,
            expected_exit_code,
        );
    }
}

pub fn test_backend(
    backend: Backend,
    workspace: &Workspace,
    filename: &str,
    input: &Input,
    use_jit: bool,
    expected_exit_code: i64,
) {
    let prefix = if let Backend::Cranelift = backend {
        "cl"
    } else {
        "llvm"
    };
    let obj_filename = format!("../test-{}-{}.obj", prefix, filename);
    let obj_path = Path::new(&obj_filename);
    if use_jit {
        if let Backend::Cranelift = backend {
            let generator = Generator::new(workspace, input, "".to_string(), use_jit);

            let result = generator.compile_nodes(obj_path);
            assert!(
                result.contains(&expected_exit_code),
                "expected main() to return {:?}, got {:?}",
                expected_exit_code,
                result.expect("main() returned nothing")
            );
        }
    } else {
        if let Backend::Cranelift = backend {
            let generator = Generator::new(workspace, input, "".to_string(), use_jit);
            generator.compile_nodes(obj_path);
        } else {
            llvm::compile(input, use_jit, obj_path);
        }
        let exe_filename = format!("../test-{}-{}.exe", prefix, filename);
        let exe_path = Path::new(&exe_filename);
        link(workspace, &obj_filename, &exe_filename, "../");
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
fn array() {
    test("array", Test::AotAndJit, "", 0, 0);
}
#[test]
fn boolean() {
    test("boolean", Test::AotAndJit, "", 0, 6);
}
#[test]
fn fibonacci() {
    test("fibonacci", Test::AotAndJit, FIBONACCI_TREE, 12, 13)
}
#[test]
fn if_else() {
    test("if_else", Test::AotAndJit, "", 0, 61);
}
#[test]
fn multiple() {
    test("multiple", Test::AotAndJit, "", 0, 10);
}
#[test]
fn negation() {
    test("negation", Test::AotAndJit, "", 0, 0);
}
#[test]
fn overload() {
    test("overload", Test::AotAndJit, "", 0, 10);
}
#[test]
fn pointer() {
    test("pointer", Test::AotAndJit, "", 0, 158);
}
#[test]
fn polystructs() {
    test("polystructs", Test::AotAndJit, "", 0, 12);
}
#[test]
fn string() {
    test("string", Test::WithPrelude, "", 0, 11);
}
#[test]
fn structs() {
    test("struct", Test::AotAndJit, STRUCTS_TREE, 24, 7);
}
#[test]
fn unicode() {
    test("unicode", Test::AotAndJit, "", 0, 114);
}
#[test]
fn while_loop() {
    test("loop", Test::AotAndJit, "", 0, 35);
}
#[test]
fn xorshift() {
    test("xorshift", Test::AotAndJit, "", 0, 6515429219844733763);
}
