#![feature(map_try_insert)]
#![feature(option_result_contains)]

use crate::{
    link::{get_module_dir, link, set_default_absolute_module_path},
    workspace::Workspace,
};
use std::{collections::HashSet, env, path::Path, time::Instant};

pub mod analyze;
mod builtin;
mod link;
mod output;
mod parse;
mod tests;
mod tokenize;
mod translate;
mod typecheck;
mod types;
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

    let mut workspace = Workspace::new();

    println!("--- BEGIN PARSE");
    let start = Instant::now();
    let mut parser = parse::Parser::new(&mut workspace);
    parser.add_module(
        parse::ModuleKind::Prelude,
        "".to_string(),
        None,
        get_module_dir().join("Prelude.hb"),
    );
    parser.add_module(
        parse::ModuleKind::Entry,
        "".to_string(),
        None,
        src_filename.into(),
    );
    parser.parse();
    let t_parse = start.elapsed();
    let mut tree = parser.tree();
    if workspace.has_errors() {
        workspace.print_errors();
        return;
    }
    if args.len() == 3 && flags.contains("-p") {
        println!("{:#?}", tree);
        return;
    }
    println!("--- END PARSE\n");

    println!("--- BEGIN ANALYZE");
    let start = Instant::now();
    let mut analyzer = analyze::Analyzer::new(&mut workspace, &tree);
    analyzer.resolve().ok();
    let t_analyze = start.elapsed();
    // print!("{}", analyzer);
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
        typecheck::Typechecker::new(&mut workspace, &mut tree, &mut definitions, &overload_sets);
    typechecker.typecheck();
    let t_typecheck = start.elapsed();
    typechecker.print();
    let (types, type_parameters) = typechecker.results();
    dbg!(&type_parameters);
    println!("{:#?}", tree);
    if workspace.has_errors() {
        workspace.print_errors();
        return;
    }
    println!("--- END TYPECHECK\n");

    let input = translate::input::Input::new(&tree, &definitions, &types, type_parameters);

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

    println!("Parse    {:>9.1?} ms", t_parse.as_secs_f64() * 1000.0);
    println!("Analyze  {:>9.1?} ms", t_analyze.as_secs_f64() * 1000.0);
    println!("Typecheck{:>9.1?} ms", t_typecheck.as_secs_f64() * 1000.0);
    println!("Generate {:>9.1?} ms", t_generate.as_secs_f64() * 1000.0);
    let t_total = t_parse + t_analyze + t_typecheck + t_generate;
    println!("Total    {:>9.1?} ms", t_total.as_secs_f64() * 1000.0);
}
