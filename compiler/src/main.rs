#![feature(map_try_insert)]
#![feature(option_result_contains)]

use color_eyre::eyre::{eyre, Result, WrapErr};
use std::{collections::HashSet, path::Path, time::Instant};

pub mod analyze;
mod builtin;
mod translate;
#[macro_use]
mod error;
mod link;
mod output;
mod parse;
mod tests;
mod tokenize;
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
    let flags: HashSet<&String> = HashSet::from_iter(args.iter().skip(2));
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

    println!("--- BEGIN TOKENIZE");
    let start = Instant::now();
    let mut tokenizer = tokenize::Tokenizer::new(&source);
    let tokens = tokenizer.tokenize();
    let t_tokenize = start.elapsed();
    tokenize::print(&source, &tokens);
    println!("--- END TOKENIZE\n");

    if args.len() == 3 && args[2] == "-t" {
        return Ok(());
    }

    println!("--- BEGIN PARSE");
    let start = Instant::now();
    let mut parser = parse::Parser::new(&source, tokens);
    parser.parse().wrap_err("Parsing error")?;
    let tree = parser.tree();
    let t_parse = start.elapsed();
    println!("{}", tree);
    println!("--- END PARSE\n");

    if args.len() == 3 && args[2] == "-p" {
        return Ok(());
    }

    println!("--- BEGIN ANALYZE");
    let start = Instant::now();
    let mut analyzer = analyze::Analyzer::new(&tree);
    analyzer.resolve().wrap_err("Name resolution error")?;
    let t_analyze = start.elapsed();
    print!("{}", analyzer);
    let mut definitions = analyzer.definitions;
    println!("--- END ANALYZE\n");

    println!("--- BEGIN TYPECHECK");
    let start = Instant::now();
    let mut typechecker = typecheck::Typechecker::new(&tree, &mut definitions);
    typechecker.check().wrap_err("Type error")?;
    let t_typecheck = start.elapsed();
    typechecker.print();
    let (types, node_types, type_parameters) = typechecker.results();
    println!("--- END TYPECHECK\n");

    let input =
        translate::input::Input::new(&tree, &definitions, &types, &node_types, type_parameters);

    println!("--- BEGIN GENERATE");
    let start = Instant::now();
    let use_jit = flags.contains(&"-j".to_string());
    let use_llvm = !use_jit && flags.contains(&"-r".to_string());
    if use_llvm {
        translate::llvm::compile(
            &input,
            use_jit,
            Path::new(&format!("llvm{}", &obj_filename)),
        );
    } else {
        let generator =
            translate::cranelift::Generator::new(&input, "object_file".to_string(), use_jit);
        generator.compile_nodes(Path::new(&obj_filename));
    }
    let t_generate = start.elapsed();
    println!("--- END GENERATE\n");

    if !use_jit {
        println!("--- BEGIN LINK [host: {}]", target_lexicon::HOST);
        if use_llvm {
            link::link(
                &format!("llvm{}", &obj_filename),
                &format!("llvm{}", &exe_filename),
                "",
            );
        } else {
            link::link(&obj_filename, &exe_filename, "");
        }
        println!("--- END LINK\n");
    }

    println!("Tokenize {:>9.1?} ms", t_tokenize.as_secs_f64() * 1000.0);
    println!("Parse    {:>9.1?} ms", t_parse.as_secs_f64() * 1000.0);
    println!("Analyze  {:>9.1?} ms", t_analyze.as_secs_f64() * 1000.0);
    println!("Typecheck{:>9.1?} ms", t_typecheck.as_secs_f64() * 1000.0);
    println!("Generate {:>9.1?} ms", t_generate.as_secs_f64() * 1000.0);
    let t_total = t_tokenize + t_parse + t_analyze + t_typecheck + t_generate;
    println!("Total    {:>9.1?} ms", t_total.as_secs_f64() * 1000.0);

    Ok(())
}
