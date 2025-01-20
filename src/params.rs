use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
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
       let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).unwrap(); // 按名称获取张量
            let shape = tensor.shape().to_vec(); // 获取张量的形状
            let data = tensor.data(); // 获取张量的数据（&[u8]）
            let data: Vec<f32> = data
                .chunks_exact(4) // 将字节数据按 4 字节（f32 的大小）分块
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())) // 将字节转换为 f32
                .collect(); // 收集为 Vec<f32>
            Tensor::new(data,&shape) // 构建 Tensor 对象
        };

        // 提取 embedding_table
        let embedding_table = get_tensor("embedding_table");

        // 提取 decoder layer 参数
        let mut rms_att_w = Vec::new();
        let mut wq = Vec::new();
        let mut wk = Vec::new();
        let mut wv = Vec::new();
        let mut wo = Vec::new();
        for i in 0..config.num_hidden_layers {
            rms_att_w.push(get_tensor(&format!("layers.{}.attention_norm.weight", i)));
            wq.push(get_tensor(&format!("layers.{}.attention.wq.weight", i)));
            wk.push(get_tensor(&format!("layers.{}.attention.wk.weight", i)));
            wv.push(get_tensor(&format!("layers.{}.attention.wv.weight", i)));
            wo.push(get_tensor(&format!("layers.{}.attention.wo.weight", i)));
        }

        // 提取 ffn layer 参数
        let mut rms_ffn_w = Vec::new();
        let mut w_up = Vec::new();
        let mut w_gate = Vec::new();
        let mut w_down = Vec::new();
        for i in 0..config.num_hidden_layers {
            rms_ffn_w.push(get_tensor(&format!("layers.{}.ffn_norm.weight", i)));
            w_up.push(get_tensor(&format!("layers.{}.feed_forward.w_up.weight", i)));
            w_gate.push(get_tensor(&format!("layers.{}.feed_forward.w_gate.weight", i)));
            w_down.push(get_tensor(&format!("layers.{}.feed_forward.w_down.weight", i)));
        }

        // 提取 output 参数
        let rms_out_w = get_tensor("norm.weight");
        let lm_head = get_tensor("output.weight");

        // 构建并返回 LLamaParams 结构体
        LLamaParams {
            embedding_table,
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w,
            lm_head,
        }
    }
}
