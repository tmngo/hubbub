use crate::analyze::Definition;
use crate::parse::{Node, NodeId, Tree};
use crate::typecheck::{Type as Typ, TypeId};
use cranelift::prelude::Type;
use std::collections::HashMap;
use std::collections::HashSet;

pub struct Input<'a> {
    pub tree: &'a Tree,
    pub definitions: &'a HashMap<u32, Definition>,
    pub types: &'a Vec<Typ>,
    pub node_types: &'a Vec<usize>,
    type_parameters: HashMap<NodeId, HashSet<Vec<TypeId>>>,
}

impl<'a> Input<'a> {
    pub fn new(
        tree: &'a Tree,
        definitions: &'a HashMap<u32, Definition>,
        types: &'a Vec<Typ>,
        node_types: &'a Vec<usize>,
        type_parameters: HashMap<NodeId, HashSet<Vec<TypeId>>>,
    ) -> Self {
        Self {
            tree: &tree,
            definitions: &definitions,
            types: &types,
            node_types: &node_types,
            type_parameters,
        }
    }
}

pub struct Data<'a> {
    pub tree: &'a Tree,
    pub definitions: &'a HashMap<u32, Definition>,
    pub types: &'a Vec<Typ>,
    pub node_types: &'a Vec<usize>,
    pub type_parameters: HashMap<NodeId, HashSet<Vec<TypeId>>>,
    pub layouts: Vec<Layout>,
}

impl<'a> Data<'a> {
    pub fn new(input: Input<'a>, layouts: Vec<Layout>) -> Self {
        Self {
            tree: input.tree,
            definitions: input.definitions,
            types: input.types,
            node_types: input.node_types,
            type_parameters: input.type_parameters,
            layouts,
        }
    }
    pub fn node(&self, index: u32) -> &Node {
        self.tree.node(index)
    }
    pub fn node_index(&self, index: u32) -> u32 {
        self.tree.node_index(index)
    }
    pub fn node_lexeme_offset(&self, node: &Node, offset: i32) -> &str {
        self.tree.node_lexeme_offset(node, offset)
    }

    pub fn get_definition(&self, node_id: NodeId) -> &Definition {
        self.definitions.get(&node_id).unwrap()
    }

    pub fn get_definition_id(&self, node_id: NodeId) -> NodeId {
        if let Definition::User(def_id) = self.get_definition(node_id) {
            return *def_id;
        }
        unreachable!("failed to get user definition")
    }

    pub fn type_id(&self, node_id: NodeId) -> TypeId {
        self.node_types[node_id as usize]
    }

    pub fn node_type(&self, node_id: NodeId) -> &Typ {
        &self.types[self.node_types[node_id as usize]]
    }

    pub fn layout(&self, node_id: NodeId) -> &Layout {
        &self.layouts[self.type_id(node_id)]
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
    align: u32,
}

impl Layout {
    pub fn new(types: &Vec<Typ>, typ: &Typ, ty: Type) -> Self {
        match typ {
            Typ::Struct { fields } => {
                let mut size = 0;
                let mut offsets = Vec::new();
                let mut memory_index = Vec::new();
                for (i, &type_id) in fields.iter().enumerate() {
                    offsets.push(size as i32);
                    memory_index.push(i as u32);
                    size += sizeof(types, type_id);
                }
                Layout::new_struct(offsets, memory_index, size, ty.bytes())
            }
            _ => Layout::new_scalar(ty.bytes(), ty.bytes()),
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
}

pub fn sizeof(types: &Vec<Typ>, type_id: usize) -> u32 {
    match &types[type_id] {
        Typ::Void => 0,
        Typ::Array { .. } => 16,
        Typ::Struct { fields } => {
            let mut size = 0;
            for f in fields {
                size += sizeof(types, *f);
            }
            size
        }
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
    fn count(&self) -> usize {
        match self {
            Shape::Scalar => 0,
            Shape::Array { count, .. } => *count as usize,
            Shape::Struct { offsets, .. } => offsets.len(),
        }
    }
    pub fn offset(&self, i: usize) -> i32 {
        match self {
            Shape::Array { stride, .. } => (stride * i as u32) as i32,
            Shape::Struct { offsets, .. } => offsets[i],
            // _ => unreachable!("Shape::offset: Scalars have no fields"),
            _ => 0,
        }
    }
    fn memory_index(&self, i: usize) -> usize {
        match self {
            Shape::Array { .. } => i,
            Shape::Struct { memory_index, .. } => memory_index[i] as usize,
            _ => unreachable!("Shape::offset: Scalars have no fields"),
        }
    }
}
