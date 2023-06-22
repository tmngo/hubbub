use crate::{
    analyze::Analyzer,
    parse::{ModuleKind, NodeId, Parser},
    typecheck::Typechecker,
    types::Type,
    workspace::Workspace,
};

pub fn check_types(source: &str, expected: &[(NodeId, Type)]) {
    let mut workspace = Workspace::new();

    let mut parser = Parser::new(&mut workspace);
    parser.add_module_from_source(ModuleKind::Entry, "".to_string(), None, source.to_string());
    parser.parse();
    let mut tree = parser.tree();
    if workspace.has_errors() {
        workspace.print_errors();
        panic!("Syntax error(s)")
    }

    // println!("{:#?}", tree);

    let mut analyzer = Analyzer::new(&mut workspace, &tree);
    analyzer.resolve().ok();
    // println!("{}", &analyzer);
    let mut definitions = analyzer.definitions;
    let overload_sets = analyzer.overload_sets;
    if workspace.has_errors() {
        workspace.print_errors();
        panic!("Name resolution error(s)")
    }

    let mut typechecker =
        Typechecker::new(&mut workspace, &mut tree, &mut definitions, &overload_sets);
    typechecker.typecheck();
    // typechecker.print();
    let (types, _type_parameters) = typechecker.results();
    for (node_id, ty) in expected {
        let type_id = tree.node(*node_id).ty;
        assert_eq!(
            *ty, types[type_id],
            "---\n [{}] expected:\n{:?}\n\ngot:\n{:?}\n ---\n",
            node_id, *ty, types[type_id]
        );
    }
    // dbg!(_type_parameters);
    // let mut type_output = String::new();
    // write!(type_output, "{:#?}", tree).ok();
    // println!("{}", type_output);
    if workspace.has_errors() {
        for diagnostic in &workspace.diagnostics {
            dbg!(&diagnostic);
        }
        workspace.print_errors();
        panic!("Type error(s)")
    }
}

const T_I16: Type = Type::Numeric {
    literal: false,
    floating: false,
    signed: true,
    bytes: 2,
};
const T_I32: Type = Type::Numeric {
    literal: false,
    floating: false,
    signed: true,
    bytes: 4,
};
const T_I64: Type = Type::Numeric {
    literal: false,
    floating: false,
    signed: true,
    bytes: 8,
};

#[test]
fn type_return() {
    check_types(
        "\
main :: () -> i64
	  return 4
end
",
        &[(6, T_I64)],
    )
}

#[test]
fn type_declaration() {
    check_types(
        "\
main :: () -> i64
    x : i64 = 4
	  return x
end
        ",
        &[(5, T_I64)],
    )
}

#[test]
fn type_declaration_cast() {
    check_types(
        "\
main :: () -> i64
    x := 4
	  return x
end
        ",
        &[(5, T_I64)],
    )
}

/// 1. infer(4):              4 -> v0(i8-i64)
/// 2. check(return x, i32)   x -> v0(i8-i32)
#[test]
fn type_infer1() {
    check_types(
        "\
main :: () -> i32
    x := 4      // x :> Int8
    return x
end
        ",
        &[(7, T_I32), (10, T_I32)],
    )
}

// fn a(x: i16) {}
// fn b(x: i64) {}
// fn c() -> i32 {
//     let x = 4;
//     a(x);
//     b(x.into());
//     x.into()
// }

///
/// 1. infer(4):  4 -> v0(i8-i64)
/// 2. infer(x):  v0(i8-i64)
/// 3. lookup(a):
///   Unique?     check(x, i16) -> v0(i8-i16)
///   Overload?   
/// 4. lookup(b):
///   Unique?     check(x, i64) -> v0(i8-i16)
/// 5. check(x, i32) -> v0(i8-i16)
/// 6. finalize(v0) -> i16
///
#[test]
fn type_infer2() {
    check_types(
        "\
a :: (x: i16)
    return
end
b :: (x: i64)
    return
end
main :: () -> i32
    x := 4      // x :> Int8
    a(x)        // x <: Int32
    b(x)        // x <: Int64
    return x
end
        ",
        &[(28, T_I16), (32, T_I64), (35, T_I32)],
    )
}

#[test]
fn type_infer3() {
    check_types(
        "\
foo :: (condition: Bool, a: Int, b: Int) -> Int
    if condition
        return a
    else
        return b
    end
end

main :: () -> Int
    x := foo(!false && true || !true, 1, 3)
    y := foo(false || true && false, 4, 5)
    return x + y
end
        ",
        &[(32, T_I64), (48, T_I64), (64, T_I64)],
    )
}

#[test]
fn type_infer4() {
    check_types(
        "\
f :: (a: Int, b: Int) -> Int
    c := 2 * (a + b)
    return c
end

main :: () -> Int
    a: i64 = 3
    b: i64 = 4
    return f(a, b)
end
        ",
        &[(20, T_I64), (45, T_I64)],
    )
}

#[test]
fn type_infer5() {
    check_types(
        "\
mutate :: (ptr: Pointer{Int}, x: Int)
    ptr@ = x
    return
end

Wrapper :: struct
    ptr: Pointer{Int}
end

main :: () -> Int
    sum := 0
    value := 15
    sum = sum + value

    p := &value
    p@ = 30
    sum = sum + p@

    w: Wrapper
    w.ptr = p
    w.ptr@ = 30
    sum = sum + w.ptr@

    wptr := &w
    wptr.ptr@ = 15
    sum = sum + w.ptr@

    mutate(p, 60)
    sum = sum + value

    return sum
end
  ",
        &[],
    )
}

#[test]
fn type_infer6() {
    check_types(
        "
main :: () -> i32
    a: i64 = 0
    b: Pointer{i64} = &a
    c := Pointer{i8}(b)
    // c: Pointer{i8} = b
    return 0
end
  ",
        &[],
    )
}

#[test]
fn type_infer7() {
    check_types(
        "
Base :: #import \"Base\"

List :: struct {T}
    data: Pointer{T}
    length: Int
    capacity: Int
end

main :: () -> i32
    list: List{i32}
    voidptr := Base.alloc(24)
    list.data = Pointer{i32}(voidptr)
    return i32(list.length)
end
  ",
        &[],
    )
}

#[test]
fn type_infer8() {
    check_types(
        "
Point :: struct{T}
    x: T
    y: T
end

make-point :: (x: i32, y: i32) -> Point{i32}
  point: Point{i32}
  point.x = x
  point.y = y
  return point
end

main :: () -> i32
    p := make-point(3, 4)
    return p.x
end
  ",
        &[],
    )
}

#[test]
fn type_infer9() {
    check_types(
        "
List :: struct {T}
    data: Pointer{T}
    length: i64
    capacity: i64
end

new-list :: (capacity: i64) -> List{i64}
    list: List{i64}
    list.length = 0
    list.capacity = capacity
    return list
end

main :: () -> i32
    m := new-list(16)
    return i32(m.length)
end
  ",
        &[],
    )
}

#[test]
fn infer_parapoly() {
    check_types(
        "
identity :: {T}(x: T) -> T
    return x
end

main :: () -> i32
    return identity(i32(3))
end
",
        &[],
    )
}

#[test]
fn infer_parapoly_deref() {
    check_types(
        "
deref :: {T}(ptr: Pointer{T}) -> T
    return ptr@
end

main :: () -> i32
    // x: i32 = 1
    // return deref(&x)
    z := false
    ptr := &z
    c := deref(ptr)
    return 0
end
",
        &[],
    )
}
