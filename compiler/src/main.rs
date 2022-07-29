#![feature(map_try_insert)]
#![feature(option_result_contains)]

use color_eyre::eyre::{eyre, Result, WrapErr};
use std::path::Path;

pub mod analyze;
#[macro_use]
mod error;
mod jit;
mod link;
mod output;
mod parse;
mod tests;
mod tokenize;
mod translator;
mod typecheck;
mod utils;

fn main() -> Result<()> {
    color_eyre::config::HookBuilder::new()
        .display_env_section(false)
        // .theme(color_eyre::config::Theme::new())
        // .display_location_section(false)
        // .capture_span_trace_by_default(true)
        .install()?;
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 1 {
        println!("USAGE: hubbub.exe file");
        return Err(eyre!("USAGE: hubbub.exe file"));
    }
    let filename = &args[1];
    let src_filename = format!("examples/{}.hb", filename);
    dbg!(&src_filename);
    let obj_filename = format!("{}.obj", filename);
    let exe_filename = format!("{}.exe", filename);
    let source = std::fs::read_to_string(src_filename)?;
    println!("{:?} ({})", source.as_bytes(), source.len());

    println!("--- BEGIN TOKENIZE");
    let mut tokenizer = tokenize::Tokenizer::new(&source);
    let tokens = tokenizer.tokenize();
    println!("--- END TOKENIZE\n");

    if args.len() == 3 && args[2] == "-t" {
        return Ok(());
    }

    println!("--- BEGIN PARSE");
    let mut parser = parse::Parser::new(&source, tokens);
    parser.parse().wrap_err("Parsing error")?;
    let tree = parser.tree();
    println!("{}", tree);
    println!("--- END PARSE\n");

    if args.len() == 3 && args[2] == "-p" {
        return Ok(());
    }

    println!("--- BEGIN ANALYZE");
    let mut analyzer = analyze::Analyzer::new(&tree);
    analyzer.resolve().wrap_err("Name resolution error")?;
    println!("{}", analyzer);
    let definitions = analyzer.definitions;
    println!("--- END ANALYZE\n");

    println!("--- BEGIN TYPECHECK");
    let mut typechecker = typecheck::Typechecker::new(&tree, &definitions);
    typechecker.check().wrap_err("Type error")?;
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
    let use_jit = args.len() == 3 && args[2] == "-j";
    let generator = jit::Generator::new(&input, "object_file".to_string(), use_jit);
    generator.compile_nodes(Path::new(&obj_filename));
    println!("--- END GENERATE\n");

    if !use_jit {
        link::link(&obj_filename, &exe_filename);
    }
    Ok(())
}
