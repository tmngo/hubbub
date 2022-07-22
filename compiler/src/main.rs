#![feature(map_try_insert)]

use std::path::Path;

pub mod analyze;
mod jit;
mod link;
mod parse;
mod tests;
mod tokenize;
mod translator;
mod typecheck;
mod utils;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 1 {
        println!("USAGE: hubbub.exe file");
        return;
    }
    let filename = &args[1];
    let src_filename = format!("examples/{}.hb", filename);
    dbg!(&src_filename);
    let obj_filename = format!("{}.obj", filename);
    let exe_filename = format!("{}.exe", filename);
    let source = std::fs::read_to_string(src_filename).unwrap();
    println!("{:?} ({})", source.as_bytes(), source.len());

    println!("--- BEGIN TOKENIZE");
    let mut tokenizer = tokenize::Tokenizer::new(&source);
    let tokens = tokenizer.tokenize();
    println!("--- END TOKENIZE\n");

    if args.len() == 3 && args[2] == "-t" {
        return;
    }

    println!("--- BEGIN PARSE");
    let mut parser = parse::Parser::new(&source, tokens);
    parser.parse();
    let tree = parser.tree();
    println!("{}", tree);
    println!("--- END PARSE\n");

    if args.len() == 3 && args[2] == "-p" {
        return;
    }

    println!("--- BEGIN ANALYZE");
    let mut analyzer = analyze::Analyzer::new(&tree);
    analyzer.resolve();
    println!("{}", analyzer);
    let definitions = analyzer.definitions;
    println!("--- END ANALYZE\n");

    println!("--- BEGIN TYPECHECK");
    let mut typechecker = typecheck::Typechecker::new(&tree, &definitions);
    typechecker.check();
    typechecker.print();
    let types = typechecker.types;
    let node_types = typechecker.node_types;
    println!("--- END TYPECHECK\n");

    let input = jit::Input {
        tree: &tree,
        definitions: &definitions,
        types: &types,
        node_types: &node_types,
    };

    println!("--- BEGIN GENERATE");
    let jit_generator = jit::Generator::new(&input, "jit_file".to_string(), true);
    jit_generator.compile(Path::new("jit_file"));
    let object_generator = jit::Generator::new(&input, "object_file".to_string(), false);
    // object_generator.compile(Path::new("object_file"));
    // object_generator.compile_nodes(nodes, indices, Path::new("main.o"));
    object_generator.compile_nodes(Path::new(&obj_filename));
    println!("--- END GENERATE\n");

    link::link(&obj_filename, &exe_filename);
}
