use crate::{
    analyze::Definition,
    parse::{Node, NodeId, NodeInfo, Tag, Tree},
    typecheck::TypeParameters,
    types::{Type as Typ, TypeId},
};
use std::{collections::HashMap, fmt::Write};

pub struct Input<'a> {
    pub tree: &'a Tree,
    pub definitions: &'a HashMap<u32, Definition>,
    pub types: &'a Vec<Typ>,
    type_parameters: TypeParameters,
}

impl<'a> Input<'a> {
    pub fn new(
        tree: &'a Tree,
        definitions: &'a HashMap<u32, Definition>,
        types: &'a Vec<Typ>,
        type_parameters: TypeParameters,
    ) -> Self {
        Self {
            tree,
            definitions,
            types,
            type_parameters,
        }
    }
}

pub struct Data<'a> {
    pub tree: &'a Tree,
    pub definitions: &'a HashMap<u32, Definition>,
    pub types: &'a Vec<Typ>,
    pub type_parameters: &'a TypeParameters,
    pub active_type_parameters: Vec<&'a Vec<TypeId>>,
    pub layouts: Vec<Layout>,
}

impl<'a> Data<'a> {
    pub fn new(input: &'a Input<'a>, layouts: Vec<Layout>) -> Self {
        Self {
            tree: input.tree,
            definitions: input.definitions,
            types: input.types,
            type_parameters: &input.type_parameters,
            active_type_parameters: vec![],
            layouts,
        }
    }
    pub fn node(&self, index: u32) -> &Node {
        self.tree.node(index)
    }
    pub fn node_index(&self, index: u32) -> u32 {
        self.tree.node_index(index)
    }

    // NodeIds can correspond to multiple types.
    pub fn type_id(&self, node_id: NodeId) -> TypeId {
        self.tree.node(node_id).ty
    }

    pub fn typ(&self, node_id: NodeId) -> &Typ {
        &self.types[self.type_id(node_id)]
    }

    pub fn layout(&self, node_id: NodeId) -> &Layout {
        &self.layouts[self.type_id(node_id)]
    }

    pub fn layout2(&self, t: TypeId) -> Layout {
        match self.types[t] {
            Typ::TypeParameter { index, .. } => {
                let t = self.active_type_parameters.last().unwrap()[index];
                Layout::new(self.types, t)
            }
            _ => Layout::new(self.types, t), // _ => self.layout(node_id).clone(), // _ => unreachable!("Invalid type: {typ:?} is not a primitive Cranelift type."),
        }
    }

    /// Returns the size of a type in bytes.
    pub fn sizeof(&self, t: TypeId) -> u32 {
        match &self.types[t] {
            Typ::Void => 0,
            Typ::Array { typ, length, .. } => (self.sizeof(*typ) as usize * length) as u32,
            Typ::Struct { fields, .. } | Typ::Tuple { fields, .. } => {
                let mut size = 0;
                for f in fields {
                    size += self.sizeof(*f);
                }
                size
            }
            Typ::Numeric { bytes, .. } => *bytes as u32,
            Typ::Boolean => 1,
            Typ::TypeParameter { index, .. } => {
                self.sizeof(self.active_type_parameters.last().unwrap()[*index])
            }
            Typ::Parameter { binding, .. } => self.sizeof(*binding),
            _ => 8,
        }
    }

    pub fn alignof(&self, t: TypeId) -> u32 {
        match &self.types[t] {
            Typ::Array { typ, .. } => self.alignof(*typ),
            Typ::Struct { fields, .. } => {
                let mut align = 0;
                for f in fields {
                    let field_align = self.alignof(*f);
                    if field_align > align {
                        align = field_align
                    }
                }
                align
            }
            _ => self.sizeof(t),
        }
    }

    pub fn format_type(&self, t: TypeId) -> String {
        match &self.types[t] {
            Typ::Any => "Any".to_string(),
            Typ::Numeric {
                floating, bytes, ..
            } => if *floating {
                match *bytes {
                    4 => "f32",
                    8 => "f64",
                    _ => unreachable!(),
                }
            } else {
                match *bytes {
                    1 => "i8",
                    2 => "i16",
                    4 => "i32",
                    8 => "i64",
                    _ => unreachable!(),
                }
            }
            .to_string(),
            Typ::Pointer { typ, .. } => format!("&{}", self.format_type(*typ)),
            Typ::Void => "Void".to_string(),
            Typ::Boolean => "Bool".to_string(),
            Typ::Struct { fields, .. } => {
                let field_list = fields
                    .iter()
                    .map(|t| self.format_type(*t))
                    .collect::<Vec<String>>()
                    .join(",");
                format!("({})", field_list)
            }
            Typ::Array { typ, length, .. } => format!("[{length}]{}", self.format_type(*typ)),
            _ => unreachable!(),
        }
    }

    pub fn mangle_function_declaration(
        &self,
        node_id: NodeId,
        includes_types: bool,
        type_arguments: &[TypeId],
        argument_types: Option<&Vec<TypeId>>,
    ) -> String {
        let node = self.node(node_id);
        let NodeInfo::Prototype {
            foreign,
            foreign_name,
        } = self.tree.info.get(&node.lhs).unwrap();
        assert_eq!(node.tag, Tag::FunctionDecl);
        let mut full_name = if *foreign {
            if let Some(name) = foreign_name {
                name.clone()
            } else {
                self.tree.name(node_id).to_string()
            }
        } else {
            self.tree.node_full_name(node_id)
        };
        let prototype = self.node(node.lhs);
        if includes_types && !full_name.starts_with("Base.") && !foreign {
            if !type_arguments.is_empty() {
                // Type parameters
                let type_params = self.node(prototype.lhs);
                let has_parameters = type_params.rhs > type_params.lhs;
                if has_parameters {
                    let type_ids = self
                        .tree
                        .range(type_params)
                        .map(|i| {
                            let ni = self.node_index(i);
                            match self.typ(ni) {
                                Typ::TypeParameter { index, .. } => type_arguments[*index],
                                _ => self.type_id(ni),
                            }
                            .to_string()
                        })
                        .collect::<Vec<String>>()
                        .join(",");
                    write!(full_name, "{{{type_ids}}}").ok();
                }
                return full_name;
            }

            // Value parameters
            let parameters = self.node(self.tree.node_extra(prototype, 0));
            let has_parameters = parameters.rhs > parameters.lhs;
            if has_parameters {
                write!(full_name, "(").ok();
            }
            if let Some(type_parameters) = argument_types {
                for ti in type_parameters {
                    write!(full_name, "{},", ti).ok();
                }
            } else {
                for i in parameters.lhs..parameters.rhs {
                    let ni = self.node_index(i);
                    let typ = self.typ(ni);
                    let ti = match typ {
                        Typ::TypeParameter { index, .. } => {
                            dbg!(argument_types).as_ref().unwrap()[*index]
                        }
                        _ => self.type_id(ni),
                    };
                    write!(full_name, "{},", ti).ok();
                }
            }
            if has_parameters {
                write!(full_name, ")").ok();
            }
        }
        full_name
    }

    // fn node_type_to_offsets(&self, node_id: NodeId) -> Vec<i32> {
    //     let type_id = self.type_id(node_id);
    //     let mut offsets = Vec::new();
    //     self.type_to_offsets(type_id, &mut offsets);
    //     offsets
    // }

    // fn type_id_to_offsets(&self, type_id: TypeId) -> Vec<i32> {
    //     let mut offsets = Vec::new();
    //     self.type_to_offsets(type_id, &mut offsets);
    //     offsets
    // }

    // fn type_to_offsets(&self, type_id: usize, offsets: &mut Vec<i32>) {
    //     let t = &self.types[type_id];
    //     // dbg!(t);
    //     match t {
    //         Typ::Struct { fields } => {
    //             for type_id in fields {
    //                 self.type_to_offsets(*type_id, offsets);
    //             }
    //         }
    //         _ => {
    //             offsets.push(if let Some(last) = offsets.last() {
    //                 last + 8
    //             } else {
    //                 0
    //             });
    //         }
    //     }
    // }
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub shape: Shape,
    pub size: u32,
    pub align: u32,
}

impl Layout {
    pub fn new(types: &Vec<Typ>, t: TypeId) -> Self {
        let typ = &types[t];
        match typ {
            Typ::Struct { fields, .. } => {
                let mut size = 0;
                let mut offsets = Vec::new();
                let mut memory_index = Vec::new();
                for (i, &type_id) in fields.iter().enumerate() {
                    offsets.push(size as i32);
                    memory_index.push(i as u32);
                    size += sizeof(types, type_id);
                }
                Layout::new_struct(offsets, memory_index, size)
            }
            Typ::Array { typ, length, .. } => {
                Layout::new_array(sizeof(types, *typ), *length as u32)
            }
            _ => Layout::new_scalar(sizeof(types, t)),
        }
    }
    pub fn new_struct(offsets: Vec<i32>, memory_index: Vec<u32>, size: u32) -> Self {
        let mut align = 1;
        for i in 1..offsets.len() - 1 {
            let field_align = (offsets[i] - offsets[i - 1]) as u32;
            if field_align > align {
                align = field_align
            }
        }
        Self {
            shape: Shape::Struct {
                offsets,
                memory_index,
            },
            size,
            align,
        }
    }
    pub fn new_scalar(size: u32) -> Self {
        Layout {
            shape: Shape::Scalar,
            size,
            align: size,
        }
    }
    pub fn new_array(stride: u32, count: u32) -> Self {
        Layout {
            shape: Shape::Array { stride, count },
            size: stride * count,
            align: stride,
        }
    }
}

/// Returns the size of a type in bytes.
pub fn sizeof(types: &Vec<Typ>, type_id: usize) -> u32 {
    match &types[type_id] {
        Typ::Void => 0,
        Typ::Array { typ, length, .. } => (sizeof(types, *typ) as usize * *length) as u32,
        Typ::Struct { fields, .. } => {
            let mut size = 0;
            for f in fields {
                size += sizeof(types, *f);
            }
            size
        }
        Typ::Numeric { bytes, .. } => *bytes as u32,
        Typ::Boolean => 1,
        _ => 8,
    }
}

#[derive(Clone, Debug)]
pub enum Shape {
    Scalar,
    Array {
        stride: u32,
        count: u32,
    },
    Struct {
        // In source order
        offsets: Vec<i32>,
        memory_index: Vec<u32>,
    },
}

impl Shape {
    // fn count(&self) -> usize {
    //     match self {
    //         Shape::Scalar => 0,
    //         Shape::Array { count, .. } => *count as usize,
    //         Shape::Struct { offsets, .. } => offsets.len(),
    //     }
    // }
    pub fn offset(&self, i: usize) -> i32 {
        match self {
            Shape::Array { stride, .. } => (stride * i as u32) as i32,
            Shape::Struct { offsets, .. } => offsets[i],
            // _ => unreachable!("Shape::offset: Scalars have no fields"),
            _ => 0,
        }
    }
    // fn memory_index(&self, i: usize) -> usize {
    //     match self {
    //         Shape::Array { .. } => i,
    //         Shape::Struct { memory_index, .. } => memory_index[i] as usize,
    //         _ => unreachable!("Shape::offset: Scalars have no fields"),
    //     }
    // }
}
