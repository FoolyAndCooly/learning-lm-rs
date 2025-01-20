use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::slice::from_raw_parts;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| {
            let tensor = safetensor.tensor(name).unwrap();
            let shape = tensor.shape();
            let data = unsafe { from_raw_parts(tensor.data().as_ptr() as *const f32, shape.iter().product()) };
            let res = Tensor::new(data.to_vec(), &shape.to_vec());
            res
        };
        // println!("{:?}", safetensor.names());
        let layer = config.num_hidden_layers;
        let mut rms_att_w = Vec::with_capacity(layer);
        let mut wq = Vec::with_capacity(layer);
        let mut wk = Vec::with_capacity(layer);
        let mut wv = Vec::with_capacity(layer);
        let mut wo = Vec::with_capacity(layer);
        let mut rms_ffn_w = Vec::with_capacity(layer);
        let mut w_up = Vec::with_capacity(layer);
        let mut w_gate = Vec::with_capacity(layer);
        let mut w_down = Vec::with_capacity(layer);

        for i in 0..layer {
            rms_att_w.push(get_tensor(&format!("model.layers.{}.input_layernorm.weight", i)));
            wq.push(get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i)));
            wk.push(get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i)));
            wv.push(get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i)));
            wo.push(get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i)));
            rms_ffn_w.push(get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i)));
            w_up.push(get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i)));
            w_gate.push(get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i)));
            w_down.push(get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i)));
        }

        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }   
    } 
}
