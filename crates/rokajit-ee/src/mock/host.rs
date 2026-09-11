use super::MockEe;
use crate::host::EeHost;

impl EeHost for MockEe {
    fn get_int_config_value(&self, _name: &str, default: i32) -> i32 {
        default
    }

    fn get_string_config_value(&self, _name: &str) -> Option<String> {
        None
    }
}
