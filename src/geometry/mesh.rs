use std::{path::Path, ptr::null_mut};

use cl3::{memory::CL_MEM_READ_WRITE, types::CL_TRUE};
use obj::{ObjError, MtlLibsLoadError, Obj, ObjData, IndexTuple};
use opencl3::{memory::Buffer, command_queue::CommandQueue, context::Context};

use super::Vec3;

#[derive(Debug)]
pub enum MeshError {
    MeshLoadError(ObjError),
    MaterialLoadError(MtlLibsLoadError),
    UntriangulatedError,
}

impl From<ObjError> for MeshError {
    fn from(x: ObjError) -> Self {
        MeshError::MeshLoadError(x)
    }
}

impl From<MtlLibsLoadError> for MeshError {
    fn from(x: MtlLibsLoadError) -> Self {
        MeshError::MaterialLoadError(x)
    }
}

pub struct Mesh {
    vertices: Vec<f32>,
    tris: Vec<u32>
}

pub struct CLMesh {
    pub vertices: Buffer<f32>,
    pub triangles: Buffer<u32>
}

impl Mesh {
    pub fn from_file(filepath: &Path) -> Result<Mesh, MeshError> {
        let mut obj: Obj = Obj::load(filepath)?;
        let mut vertices: Vec<f32> = Vec::new();
        let mut tris: Vec<u32> = Vec::new();
        let obj_data: &ObjData = &obj.data;
        for chunk in obj_data.position.iter() {
            vertices.push(chunk[0]);
            vertices.push(chunk[1]);
            vertices.push(chunk[2]);
        }

        let object = &obj_data.objects[0];
        for group in object.groups.iter() {
            for poly in group.polys.iter() {
                let index_tup_vec: &Vec<IndexTuple> = &poly.0;
                if index_tup_vec.len() != 3 {
                    return Err(MeshError::UntriangulatedError);
                }

                let v0 = index_tup_vec[0];
                let v1 = index_tup_vec[1];
                let v2 = index_tup_vec[2];
                tris.push(v0.0 as u32);
                tris.push(v1.0 as u32);
                tris.push(v2.0 as u32);
            }
        }

        Ok(
            Mesh { vertices: vertices, tris: tris }
        )
    }

    pub fn to_cl_mesh(&self, context: &Context, command_queue: &CommandQueue) -> CLMesh {
        let mut vertex_buffer: Buffer<f32>;
        let mut index_buffer: Buffer<u32>;
        unsafe {
            vertex_buffer = Buffer::create(
                context, 
                CL_MEM_READ_WRITE, 
                self.vertices.len(), 
                null_mut()
            ).expect("failed to create vertex buffer");

            index_buffer = Buffer::create(
                context, 
                CL_MEM_READ_WRITE, 
                self.tris.len(), 
                null_mut()
            ).expect("failed to create index buffer");

            command_queue.enqueue_write_buffer(
                &mut vertex_buffer, 
                CL_TRUE, 
                0, 
                &self.vertices, 
                &[]
            ).expect("failed to populate vertex buffer");
            
            command_queue.enqueue_write_buffer(
                &mut index_buffer, 
                CL_TRUE, 
                0, 
                &self.tris, 
                &[]
            ).expect("failed to populate index buffer");
        }
        
        CLMesh { vertices: vertex_buffer, triangles: index_buffer }
    }
}