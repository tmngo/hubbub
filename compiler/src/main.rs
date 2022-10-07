#![feature(map_try_insert)]
#![feature(option_result_contains)]

use crate::{
    link::{link, prepend_module, set_default_absolute_module_path},
    workspace::Workspace,
};
use std::{collections::HashSet, env, fs, path::Path, time::Instant};

pub mod analyze;
mod builtin;
mod link;
mod output;
mod parse;
mod tests;
mod tokenize;
mod translate;
mod typecheck;
mod utils;
mod workspace;

fn main() {
    set_default_absolute_module_path();
    let args: Vec<String> = env::args().collect();
    let flags: HashSet<&str> = HashSet::from_iter(args.iter().skip(2).map(|s| s.as_str()));
    if args.len() == 1 {
        println!("USAGE: hubbub.exe file");
        return;
    }
    let filename = &args[1];
    let src_filename = format!("{}.hb", filename);
    let obj_filename = format!("{}.obj", filename);
    let exe_filename = format!("{}.exe", filename);
    let source = prepend_module("Prelude.hb", &fs::read_to_string(&src_filename).unwrap());

    let mut workspace = Workspace::new();

    workspace.files.add(src_filename, source.clone());

    println!("--- BEGIN TOKENIZE");
    let start = Instant::now();
    let mut tokenizer = tokenize::Tokenizer::new(&source);
    let tokens = tokenizer.tokenize();
    let t_tokenize = start.elapsed();
    tokenize::print(&source, &tokens);
    println!("--- END TOKENIZE\n");

    if args.len() == 3 && flags.contains("-t") {
        return;
    }

    println!("--- BEGIN PARSE");
    let start = Instant::now();
    let mut parser = parse::Parser::new(&mut workspace, &source, tokens);
    parser.parse();
    let t_parse = start.elapsed();
    let tree = parser.tree();
    println!("{}", tree);
    if workspace.has_errors() {
        workspace.print_errors();
        return;
    }
    println!("--- END PARSE\n");

    if args.len() == 3 && flags.contains("-p") {
        return;
    }

    println!("--- BEGIN ANALYZE");
    let start = Instant::now();
    let mut analyzer = analyze::Analyzer::new(&mut workspace, &tree);
    analyzer.resolve().ok();
    let t_analyze = start.elapsed();
    print!("{}", analyzer);
    let mut definitions = analyzer.definitions;
    let overload_sets = analyzer.overload_sets;
    if workspace.has_errors() {
        workspace.print_errors();
        return;
    }
    println!("--- END ANALYZE\n");

    println!("--- BEGIN TYPECHECK");
    let start = Instant::now();
    let mut typechecker =
        typecheck::Typechecker::new(&mut workspace, &tree, &mut definitions, &overload_sets);
    typechecker.check().ok();
    let t_typecheck = start.elapsed();
    typechecker.print();
    let (types, node_types, type_parameters) = typechecker.results();
    if workspace.has_errors() {
        workspace.print_errors();
        return;
    }
    println!("--- END TYPECHECK\n");

    let input =
        translate::input::Input::new(&tree, &definitions, &types, &node_types, type_parameters);

    println!("--- BEGIN GENERATE");
    let start = Instant::now();
    let use_jit = flags.contains("-j");
    let use_llvm = !use_jit && flags.contains("-r");
    if use_llvm {
        translate::llvm::compile(
            &input,
            use_jit,
            Path::new(&format!("llvm{}", &obj_filename)),
        );
    } else {
        let generator = translate::cranelift::Generator::new(
            &workspace,
            &input,
            "object_file".to_string(),
            use_jit,
        );
        generator.compile_nodes(Path::new(&obj_filename));
    }
    let t_generate = start.elapsed();
    println!("--- END GENERATE\n");

    if !use_jit {
        println!("--- BEGIN LINK [host: {}]", target_lexicon::HOST);
        if use_llvm {
            link(
                &workspace,
                &format!("llvm{}", &obj_filename),
                &format!("llvm{}", &exe_filename),
                "",
            );
        } else {
            link(&workspace, &obj_filename, &exe_filename, "");
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
}
