use crate::workspace::Workspace;
use std::io::{self, Write};
use std::process::Command;
use target_lexicon::Triple;

/**
 * cl.exe /Fe:main filename.obj /Fe:filename.exe /link /defaultlib:libcmt
 * link.exe filename.obj /out:filename.exe /defaultlib:libcmt
 *
 * clang -o a.exe filename
 * link -out:a.exe
 *
 *
 *
 *
 */

pub fn link(workspace: &Workspace, object_filename: &str, output_filename: &str, base_dir: &str) {
    let host = &format!("{}", Triple::host());
    let tool = cc::Build::new()
        .host(host)
        .target(host)
        .opt_level(0)
        .get_compiler();
    let mut command = tool.to_command();

    let compiler_path = std::env::current_exe().unwrap();
    let compiler_dir = compiler_path.parent().unwrap().to_path_buf();
    let runtime_path = compiler_dir
        .join(base_dir)
        .join("hubbub_runtime.lib")
        .into_os_string()
        .into_string()
        .unwrap();

    let static_library_names = [
        "advapi32", "bcrypt", "kernel32", "userenv", "ws2_32", "msvcrt",
    ];

    // Static libs are found via absolute path or in PATH.
    // Dynamic libs are found next to the executable or in PATH.
    let mut args = vec![];
    if tool.is_like_msvc() {
        args.extend([
            format!("-Fe{}", output_filename),
            object_filename.to_string(),
            runtime_path,
        ]);
        args.extend(
            static_library_names
                .iter()
                .map(|name| format!("{}.lib", name)),
        );
        args.extend(workspace.library_files.iter().map(|file| {
            compiler_dir
                .join(file.as_str())
                .with_extension("lib")
                .into_os_string()
                .into_string()
                .unwrap()
        }));
    } else if tool.is_like_clang() {
        args.extend([
            format!("-o{}", output_filename),
            object_filename.to_string(),
            runtime_path,
        ]);
        args.extend(
            static_library_names
                .iter()
                .map(|name| format!("-l{}", name)),
        );
    } else {
        println!("Could not find clang or cl.");
        return;
    };
    command.args(args);

    println!("Command: {:?}", command);

    let output = command.output().expect("failed to link");
    println!("Status:  {}", output.status);
    io::stdout().write_all(&output.stdout).unwrap();
    io::stderr().write_all(&output.stderr).unwrap();
    assert!(output.status.success());
}

// #[cfg(not(windows))]
pub fn link_gcc(object_filename: &str, output_filename: &str) {
    let mut command = Command::new("gcc");
    command.args([
        &format!("-o{}", output_filename),
        object_filename,
        "target/debug/libhubbub_runtime.a",
        "-ldl",
        "-pthread",
    ]);

    println!("Command: {:?}", command);

    let output = command.output().expect("failed to link");
    println!("Status:  {}", output.status);
    io::stdout().write_all(&output.stdout).unwrap();
    io::stderr().write_all(&output.stderr).unwrap();
    assert!(output.status.success());
}

pub fn set_default_absolute_module_path() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let modules_path = std::path::Path::new(&manifest_dir).with_file_name("modules");
    std::env::set_var("ABSOLUTE_MODULE_PATH", modules_path);
}
