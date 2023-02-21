use crate::{
    analyze::Definition,
    parse::{Node, NodeId, NodeInfo, Tag, Tree},
    types::{Type as Typ, TypeId},
};
use std::{collections::HashMap, fmt::Write};

pub struct Input<'a> {
    pub tree: &'a Tree,
    pub definitions: &'a HashMap<u32, Definition>,
    pub types: &'a Vec<Typ>,
    type_parameters: HashMap<NodeId, HashMap<Vec<TypeId>, Vec<TypeId>>>,
}

impl<'a> Input<'a> {
    pub fn new(
        tree: &'a Tree,
        definitions: &'a HashMap<u32, Definition>,
        types: &'a Vec<Typ>,
        type_parameters: HashMap<NodeId, HashMap<Vec<TypeId>, Vec<TypeId>>>,
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
    pub type_parameters: &'a HashMap<NodeId, HashMap<Vec<TypeId>, Vec<TypeId>>>,
    pub active_type_parameters: Option<&'a Vec<TypeId>>,
    pub layouts: Vec<Layout>,
}

impl<'a> Data<'a> {
    pub fn new(input: &'a Input<'a>, layouts: Vec<Layout>) -> Self {
        Self {
            tree: input.tree,
            definitions: input.definitions,
            types: input.types,
            type_parameters: &input.type_parameters,
            active_type_parameters: None,
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

    pub fn mangle_function_declaration(
        &self,
        node_id: NodeId,
        includes_types: bool,
        type_parameters: Option<&Vec<TypeId>>,
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
            let parameters = self.node(self.tree.node_extra(prototype, 0));
            if parameters.rhs > parameters.lhs {
                write!(full_name, "|").ok();
            }
            if let Some(type_parameters) = type_parameters {
                for ti in type_parameters {
                    write!(full_name, "{},", ti).ok();
                }
            } else {
                for i in parameters.lhs..parameters.rhs {
                    let ni = self.node_index(i);
                    let typ = self.typ(ni);
                    let ti = match typ {
                        Typ::Parameter { index, .. } => {
                            dbg!(type_parameters).as_ref().unwrap()[*index]
                        }
                        _ => self.type_id(ni),
                    };
                    write!(full_name, "{},", ti).ok();
                }
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
    pub fn new(types: &Vec<Typ>, typ: &Typ, bytes: u32) -> Self {
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
                Layout::new_struct(offsets, memory_index, size, bytes)
            }
            Typ::Array { typ, length, .. } => {
                Layout::new_array(sizeof(types, *typ), *length as u32)
            }
            _ => Layout::new_scalar(bytes, bytes),
        }
    }
    pub fn new_struct(offsets: Vec<i32>, memory_index: Vec<u32>, size: u32, align: u32) -> Self {
        Self {
            shape: Shape::Struct {
                offsets,
                memory_index,
            },
            size,
            align,
        }
    }
    pub fn new_scalar(size: u32, align: u32) -> Self {
        Layout {
            shape: Shape::Scalar,
            size,
            align,
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
