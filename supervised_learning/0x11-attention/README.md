# Attention

[Transformer](https://www.tensorflow.org/tutorials/text/transformer)

### Tasks

#### [RNN Encoder](./0-rnn_encoder.py)
- Create a class RNNEncoder that inherits from tensorflow.keras.layers.Layer to encode for machine translation

#### [Self Attention](./1-self_attention.py)
- Create a class SelfAttention that inherits from tensorflow.keras.layers.Layer to calculate the attention for machine translation based on [this paper](https://arxiv.org/pdf/1409.0473.pdf)

#### [RNN Decoder](./2-rnn_decoder.py)
- Create a class RNNDecoder that inherits from tensorflow.keras.layers.Layer to decode for machine translation

#### [Positional Encoding](./4-positional_encoding.py)
- Write the function def positional_encoding(max_seq_len, dm): that calculates the positional encoding for a transformer

#### [Scaled Dot Product Attention](./5-sdp_attention.py)
![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/7/8f5aadef511d9f646f5009756035b472073fe896.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210104%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210104T143003Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=bf1c78cbc956e0ff36b5687a78c1c63a30612e335d457e5a109da45e21932e4d)
- Write the function def sdp_attention(Q, K, V, mask=None) that calculates the scaled dot product attention

#### [Multi Head Attention](./6-multihead_attention.py)
![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/7/4a5aaa54ebdc32529b4f09a5f22789dc267e0796.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210104%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210104T143003Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=dff713e7d36bec36a505208c2b3f031b7e19c07a3e6ea638ed02c27599a8f46e)
- Create a class MultiHeadAttention that inherits from tensorflow.keras.layers.Layer to perform multi head attention

#### [Transformer Encoder Block](./7-transformer_encoder_block.py)
![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/7/50a5309eae279760a5d6fc6031aa045eafd0e605.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210104%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210104T143003Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e617938855275fb39b58c7b7de98605ed5e7d947561bfbbbfaa764f9d1a2ccff)
- Create a class EncoderBlock that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer

#### [Transformer Decoder Block](./8-transformer_decoder_block.py)
- Create a class DecoderBlock that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer

#### [Transformer Encoder](./9-transformer_encoder.py)
- Create a class Encoder that inherits from tensorflow.keras.layers.Layer to create the encoder for a transformer

#### [Transformer Decoder](./10-transformer_decoder.py)
- Create a class Decoder that inherits from tensorflow.keras.layers.Layer to create the decoder for a transformer

#### [Transformer Network](./11-transformer.py)
-Create a class Transformer that inherits from tensorflow.keras.Model to create a transformer network