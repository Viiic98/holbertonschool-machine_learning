# Attention

![a](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/Feeding-Hidden-State-as-Input-to-Decoder.png)
## Tasks

### [RNN Encoder](./0-rnn_encoder.py)
- Create a class RNNEncoder that inherits from tensorflow.keras.layers.Layer to encode for machine translation

### [Self Attention](./1-self_attention.py)
- Create a class SelfAttention that inherits from tensorflow.keras.layers.Layer to calculate the attention for machine translation based on [this paper](https://arxiv.org/pdf/1409.0473.pdf)

### [RNN Decoder](./2-rnn_decoder.py)
- Create a class RNNDecoder that inherits from tensorflow.keras.layers.Layer to decode for machine translation

### [Positional Encoding](./4-positional_encoding.py)
- Write the function def positional_encoding(max_seq_len, dm): that calculates the positional encoding for a transformer

### [Scaled Dot Product Attention](./5-sdp_attention.py)
- Write the function def sdp_attention(Q, K, V, mask=None) that calculates the scaled dot product attention

### [Multi Head Attention](./6-multihead_attention.py)
- Create a class MultiHeadAttention that inherits from tensorflow.keras.layers.Layer to perform multi head attention

### [Transformer Encoder Block](./7-transformer_encoder_block.py)
- Create a class EncoderBlock that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer

### [Transformer Decoder Block](./8-transformer_decoder_block.py)
- Create a class DecoderBlock that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer

### [Transformer Encoder](./9-transformer_encoder.py)
- Create a class Encoder that inherits from tensorflow.keras.layers.Layer to create the encoder for a transformer

### [Transformer Decoder](./10-transformer_decoder.py)
- Create a class Decoder that inherits from tensorflow.keras.layers.Layer to create the decoder for a transformer

### [Transformer Network](./11-transformer.py)
-Create a class Transformer that inherits from tensorflow.keras.Model to create a transformer network