use super::MockEe;
use crate::ee_info::{BoundaryMap, DebugInfo, NativeVarInfo};
use crate::handles::MethodHandle;

impl DebugInfo for MockEe {
    fn get_boundaries(&self, _ftn: MethodHandle) -> Vec<u32> {
        Vec::new()
    }

    fn set_boundaries(&self, _ftn: MethodHandle, map: &[BoundaryMap]) {
        self.sink_log
            .borrow_mut()
            .push(format!("set_boundaries({})", map.len()));
    }

    fn get_vars(&self, _ftn: MethodHandle) -> Vec<u32> {
        Vec::new()
    }

    fn set_vars(&self, _ftn: MethodHandle, vars: &[NativeVarInfo]) {
        self.sink_log
            .borrow_mut()
            .push(format!("set_vars({})", vars.len()));
    }
}
