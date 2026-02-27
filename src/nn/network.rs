use crate::nn::layer::Layer;

#[derive(Default)]
pub struct Network {
    layers: Vec<Layer>
}

#[derive(Default)]
pub struct NetworkBuilder {
    layers: Vec<Layer>
}

impl Network {
    pub fn builder() -> NetworkBuilder {
        NetworkBuilder::default()
    }
}

impl NetworkBuilder {
    pub fn add_layer(mut self, layer: Layer) -> Result<(), String> {
        self.layers.push(layer);
        Ok(())
    }

    pub fn build(self) -> Result<Network, String> {
        if self.layers.is_empty() {
            return Err(String::from("Failed to build network: network must have at least one layer"));
        }
        Ok(Network { layers: self.layers })
    }
}