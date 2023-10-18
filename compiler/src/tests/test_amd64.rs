use crate::{link::link, workspace::Workspace};
use cranelift::prelude::{isa::lookup, settings};
use cranelift_module::Linkage;
use iced_x86::{code_asm::*, IcedError};
use object::{
    write::{Object, StandardSection, Symbol, SymbolSection},
    Architecture, BinaryFormat, Endianness, SymbolFlags, SymbolKind, SymbolScope,
};

/*
    0: 55                             push    rbp
    1: 48 89 e5                       mov     rbp, rsp
    4: 48 83 ec 10                    sub     rsp, 16
    8: 48 0f be 15 18 00 00 00        movsx   rdx, byte ptr [rip + 24] # 0x28 <main+0x28>
    10: 4c 8d 04 24                   lea     r8, [rsp]
    14: 49 89 10                      mov     qword ptr [r8], rdx
    17: 4c 8d 04 24                   lea     r8, [rsp]
    1b: 49 8b 00                      mov     rax, qword ptr [r8]
    1e: 48 83 c4 10                   add     rsp, 16
    22: 48 89 ec                      mov     rsp, rbp
    25: 5d                            pop     rbp
    26: c3                            ret
    27: 00 20                         add     byte ptr [rax], ah
    29: 00 00                         add     byte ptr [rax], al
    2b: 00 00                         add     byte ptr [rax], al
    2d: 00 00                         add     byte ptr [rax], al
    2f: 00                            <unknown>
*/

#[test]
pub fn amd() -> Result<(), IcedError> {
    // set_default_absolute_module_path();

    // Encode instructions.
    let mut a = CodeAssembler::new(64)?;

    a.push(rbp)?;
    a.mov(rbp, rsp)?;
    a.sub(rsp, 16)?;
    let mut ip_label = a.create_label();
    a.movsx(rdx, byte_ptr(ip_label))?;
    a.lea(r8, ptr(rsp))?;
    a.mov(qword_ptr(r8), rdx)?;
    a.lea(r8, ptr(rsp))?;
    a.mov(rax, qword_ptr(r8))?;
    a.add(rsp, 16)?;
    a.mov(rsp, rbp)?;
    a.pop(rbp)?;
    a.ret()?;
    a.db(b"\x00")?;
    a.set_label(&mut ip_label)?;
    a.zero_bytes()?;
    a.db(b"\x20\x00\x00\x00\x00\x00\x00\x00")?;

    let bytes = a.assemble(0x0)?;

    // Write object file.

    let flag_builder = settings::builder();
    let isa_builder = lookup(target_lexicon::HOST).unwrap();
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .unwrap();
    let align = (isa.function_alignment() as u64).max(isa.symbol_alignment());

    let mut object = Object::new(BinaryFormat::Coff, Architecture::X86_64, Endianness::Little);
    object.add_file_symbol("object_file".into());

    let (scope, weak) = translate_linkage(Linkage::Export);

    let section = object.section_id(StandardSection::Text);
    let symbol = Symbol {
        name: "main".as_bytes().to_vec(),
        value: 0,
        size: 0,
        kind: SymbolKind::Text,
        scope,
        weak,
        section: SymbolSection::Undefined,
        flags: SymbolFlags::None,
    };
    let symbol_id = object.add_symbol(symbol);
    object.add_symbol_data(symbol_id, section, &bytes, align);

    let obj_bytes = object.write().unwrap();
    std::fs::write("../amd.obj", obj_bytes).unwrap();

    let workspace = Workspace::new();
    link(&workspace, "../amd.obj", "../amd.exe", "../");

    Ok(())
}

fn translate_linkage(linkage: Linkage) -> (SymbolScope, bool) {
    let scope = match linkage {
        Linkage::Import => SymbolScope::Unknown,
        Linkage::Local => SymbolScope::Compilation,
        Linkage::Hidden => SymbolScope::Linkage,
        Linkage::Export | Linkage::Preemptible => SymbolScope::Dynamic,
    };
    // TODO: this matches rustc_codegen_cranelift, but may be wrong.
    let weak = linkage == Linkage::Preemptible;
    (scope, weak)
}
