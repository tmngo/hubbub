use crate::workspace::Workspace;
use std::{env, path, process::Command};
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

    let compiler_path = env::current_exe().unwrap();
    let compiler_dir = compiler_path.parent().unwrap().to_path_buf();
    let runtime_path = compiler_dir
        .join(base_dir)
        .join("hubbub_runtime.lib")
        .into_os_string()
        .into_string()
        .unwrap();

    // rustflags = ["-C", "target-feature=+crt-static", "--print=native-static-libs"]
    let static_library_names = [
        "advapi32",
        "bcrypt",
        "kernel32",
        "userenv",
        "ws2_32",
        "msvcrt",
        "ntdll",
        "legacy_stdio_definitions",
        "opengl32",
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
        args.extend(
            workspace
                .library_files
                .iter()
                .filter(|file| !static_library_names.contains(&file.as_str()))
                .map(|file| {
                    compiler_dir
                        .join(file.as_str())
                        .with_extension("dll.lib")
                        .into_os_string()
                        .into_string()
                        .unwrap()
                }),
        );
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
    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success());
}

// #[cfg(not(windows))]
#[allow(dead_code)]
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
    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success());
}

#[cfg(test)]
/// Look for the module directory relative to the cargo manifest.
pub fn get_module_dir() -> path::PathBuf {
    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        path::Path::new(&manifest_dir).with_file_name("modules")
    } else {
        panic!("Failed to get test module directory!")
    }
}

#[cfg(not(test))]
/// Look for the module directory next to the compiler exe.
pub fn get_module_dir() -> path::PathBuf {
    env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .join("modules")
}
