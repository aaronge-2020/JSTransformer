const tf = await import("https://esm.sh/@tensorflow/tfjs@4.10.0");

function positionalEncoding(length, depth) {
  /*
    Generates the positional encoding for a transformer model.

    Args:
        length: Sequence length
        depth: Dimensions of the embedding vector

    */
  depth = depth / 2;

  const positions = tf.range(0, length).reshape([-1, 1]); // (seq, 1)
  const depths = tf.range(0, depth).div(depth).reshape([1, -1]); // (1, depth)

  const angleRates = tf.pow(10000, depths.mul(-1)); // (1, depth)
  const angleRads = positions.mul(angleRates); // (pos, depth)

  const sinAngleRads = tf.sin(angleRads);
  const cosAngleRads = tf.cos(angleRads);

  const posEncoding = tf.concat([sinAngleRads, cosAngleRads], 1); // Concatenate along axis -1 (1 in JS)

  return posEncoding.toFloat();
}

class PositionalEmbedding extends tf.layers.Layer {
  constructor(vocabSize, dModel, config) {
    super(config);
    this.dModel = dModel;
    this.embedding = tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: dModel,
      maskZero: true,
    });
    this.posEncoding = positionalEncoding(vocabSize, dModel);
  }

  call(inputs) {
    let input = inputs;
    if (Array.isArray(input)) {
      input = input[0];
    }

    const length = input.shape[1];
    let x = this.embedding.apply(input);

    // This factor sets the relative scale of the embedding and positionalEncoding
    x = x.mul(tf.sqrt(tf.scalar(this.dModel, "float32")));

    // We need to slice the positional encoding, because the original shape is [2048, 512], where 2048 is the max sequence length and 512 is the embedding size
    const posEncodingSliced = this.posEncoding.slice([0, 0], [length, -1]);
    x = x.add(posEncodingSliced);

    return x;
  }

  getClassName() {
    return "PositionalEmbedding";
  }
}
class MultiHeadAttention extends tf.layers.Layer {
  constructor(d_model, num_heads, config, causal = false) {
    super(config);
    this.num_heads = num_heads;
    this.d_model = d_model;
    this.causal = causal; // Add this line to include a causal flag
    if (d_model % this.num_heads !== 0) {
      throw new Error("d_model must be divisible by num_heads");
    }
    this.depth = Math.floor(d_model / this.num_heads);

    this.wq = tf.layers.dense({ units: d_model });
    this.wk = tf.layers.dense({ units: d_model });
    this.wv = tf.layers.dense({ units: d_model });
    this.dense = tf.layers.dense({ units: d_model });
  }

  scaledDotProductAttention(q, k, v) {
    const matmulQK = tf.matMul(q, k.transpose([0, 1, 3, 2]));
    const dk = k.shape[k.shape.length - 1];
    let scaledAttentionLogits = matmulQK.div(tf.sqrt(dk));

    // Apply causal mask if required
    if (this.causal) {
      const seqLen =
        scaledAttentionLogits.shape[scaledAttentionLogits.shape.length - 2];
      const upperTriangular = tf.linalg.bandPart(
        tf.ones([seqLen, seqLen]),
        0,
        -1
      );
      const identityMatrix = tf.eye(seqLen);
      const causalMask = upperTriangular.sub(identityMatrix);
      scaledAttentionLogits = scaledAttentionLogits.add(
        causalMask.mul(tf.scalar(-1e9))
      );
    }

    const attentionWeights = tf.softmax(scaledAttentionLogits, -1);
    const output = tf.matMul(attentionWeights, v);

    return [output, attentionWeights];
  }

  splitHeads(x, batch_size) {
    const reshaped = x.reshape([batch_size, -1, this.num_heads, this.depth]);
    return reshaped.transpose([0, 2, 1, 3]);
  }

  call(v, k, q) {
    const batchSize = q.shape[0];
    const qProcessed = this.wq.apply(q);
    const kProcessed = this.wk.apply(k);
    const vProcessed = this.wv.apply(v);

    const qSplit = this.splitHeads(qProcessed, batchSize);
    const kSplit = this.splitHeads(kProcessed, batchSize);
    const vSplit = this.splitHeads(vProcessed, batchSize);

    const [scaledAttention, attentionWeights] = this.scaledDotProductAttention(
      qSplit,
      kSplit,
      vSplit
    );

    const scaledAttentionTransposed = scaledAttention.transpose([0, 2, 1, 3]);
    const concatAttention = scaledAttentionTransposed.reshape([
      batchSize,
      -1,
      this.d_model,
    ]);
    const output = this.dense.apply(concatAttention);

    return [output, attentionWeights];
  }

  getClassName() {
    return "MultiHeadAttention";
  }
}

class BaseAttention extends tf.layers.Layer {
  constructor(d_model, num_heads, config, causal = false) {
    super(config);

    this.mha = new MultiHeadAttention(d_model, num_heads, config, causal);
    this.layernorm = tf.layers.layerNormalization();
    this.add = tf.layers.add();
  }

  getClassName() {
    return "BaseAttention";
  }
}

class CrossAttention extends BaseAttention {
  constructor(d_model, num_heads, config) {
    super(d_model, num_heads, config);
  }
  call(inputs, kwargs) {
    const [x, context] = inputs;

    // Forward pass through MultiHeadAttention
    const [attnOutput, attnScores] = this.mha.call(x, context, context, null);

    // Cache the attention scores for later use or plotting
    this.lastAttnScores = attnScores;

    // Add the output to the original input (Residual connection)
    const addedOutput = this.add.apply([x, attnOutput]);

    // Layer normalization
    const normalizedOutput = this.layernorm.apply(addedOutput);

    return normalizedOutput;
  }

  // Define output shape based on input shape
  computeOutputShape(inputShape) {
    return inputShape[0];
  }
}

class GlobalSelfAttention extends BaseAttention {
  constructor(d_model, num_heads, config) {
    super(d_model, num_heads, config);
  }

  call(inputs, kwargs) {
    const x = inputs;

    // Forward pass through MultiHeadAttention
    const [attnOutput, attnScores] = this.mha.call(x, x, x);

    // Add the output to the original input (Residual connection)
    const addedOutput = this.add.apply([x, attnOutput]);

    // Layer normalization
    const normalizedOutput = this.layernorm.apply(addedOutput);

    return normalizedOutput;
  }

  // Define output shape based on input shape
  computeOutputShape(inputShape) {
    return inputShape;
  }
}

class CausalSelfAttention extends BaseAttention {
  constructor(d_model, num_heads, config) {
    super(d_model, num_heads, config, true); // Enable causal attention by setting the last argument to true
  }

  call(x) {
    // Multi-head attention layer with causal mask enabled
    const [attnOutput, _] = this.mha.call(x, x, x);

    // Add & normalize layer
    const addOutput = this.add.apply([x, attnOutput]);
    const output = this.layernorm.apply(addOutput);

    return output;
  }
}
class FeedForward extends tf.layers.Layer {
  /*

    Implements the feedforward network used in the transformer.

    Args:
        dModel: Depth of the input vector
        dff: Hidden layer size
        dropoutRate: Dropout rate

    */

  constructor(dModel, dff, dropoutRate = 0.1) {
    super();
    this.seq = tf.sequential();
    this.seq.add(
      tf.layers.dense({ units: dff, activation: "relu", inputShape: [dModel] })
    );
    this.seq.add(tf.layers.dense({ units: dModel }));
    this.seq.add(tf.layers.dropout({ rate: dropoutRate }));
    this.layerNorm = tf.layers.layerNormalization();
  }

  call(x) {
    const seqOutput = this.seq.apply(x);
    x = tf.add(x, seqOutput);
    x = this.layerNorm.apply(x);
    return x;
  }

  getClassName() {
    return "FeedForward";
  }
}

class EncoderLayer extends tf.layers.Layer {
  constructor(d_model, num_heads, dff, dropout_rate = 0.1) {
    super();
    this.self_attention = new GlobalSelfAttention(d_model, num_heads);
    this.ffn = new FeedForward(d_model, dff);
  }

  call(inputs) {
    let x = this.self_attention.apply(inputs);
    x = this.ffn.apply(x);
    return x;
  }
  getClassName() {
    return "EncoderLayer";
  }
}

class Encoder extends tf.layers.Layer {
  constructor(
    num_layers,
    d_model,
    num_heads,
    dff,
    vocab_size,
    dropout_rate = 0.1
  ) {
    super();
    this.d_model = d_model;
    this.num_layers = num_layers;

    this.pos_embedding = new PositionalEmbedding(vocab_size, d_model);

    this.enc_layers = Array.from(
      { length: num_layers },
      () => new EncoderLayer(d_model, num_heads, dff, dropout_rate)
    );

    this.dropout = tf.layers.dropout({ rate: dropout_rate });
  }

  call(inputs) {
    let x = this.pos_embedding.apply(inputs); // Assuming shape `(batch_size, seq_len, d_model)`.

    // Add dropout
    x = this.dropout.apply(x);

    for (let i = 0; i < this.num_layers; i++) {
      x = this.enc_layers[i].apply(x);
    }

    return x; // Should have shape `(batch_size, seq_len, d_model)`.
  }

  getClassName() {
    return "Encoder";
  }
}

class DecoderLayer extends tf.layers.Layer {
  constructor(d_model, num_heads, dff, dropout_rate = 0.1) {
    super();
    this.causalSelfAttention = new CausalSelfAttention(
      d_model,
      num_heads,
      dropout_rate
    );

    this.crossAttention = new CrossAttention(d_model, num_heads, dropout_rate);

    this.ffn = new FeedForward(d_model, dff);
  }

  call(inputs) {
    const [x, context] = inputs;
    let out = this.causalSelfAttention.apply(x);
    out = this.crossAttention.apply([out, context]);

    // Cache the last attention scores for plotting later
    this.lastAttnScores = this.crossAttention.lastAttnScores;

    out = this.ffn.apply(out);
    return out;
  }

  getClassName() {
    return "DecoderLayer";
  }
}

class Decoder extends tf.layers.Layer {
  constructor(
    num_layers,
    d_model,
    num_heads,
    dff,
    vocab_size,
    dropout_rate = 0.1
  ) {
    super();
    this.d_model = d_model;
    this.num_layers = num_layers;

    this.pos_embedding = new PositionalEmbedding(vocab_size, d_model);
    this.dropout = tf.layers.dropout({ rate: dropout_rate });
    this.dec_layers = Array.from(
      { length: num_layers },
      () => new DecoderLayer(d_model, num_heads, dff, dropout_rate)
    );

    this.last_attn_scores = null;
  }

  call(inputs) {
    const [x, context] = inputs;

    let out = this.pos_embedding.apply(x); // Assuming shape `(batch_size, target_seq_len, d_model)`

    out = this.dropout.apply(out);

    for (let i = 0; i < this.num_layers; i++) {
      out = this.dec_layers[i].apply([out, context]);
    }

    this.last_attn_scores = this.dec_layers[this.num_layers - 1].lastAttnScores;

    // The shape of out is (batch_size, target_seq_len, d_model)
    return out;
  }

  getClassName() {
    return "Decoder";
  }
}

class Transformer extends tf.LayersModel {
  constructor(
    num_layers,
    d_model,
    num_heads,
    dff,
    input_vocab_size,
    target_vocab_size,
    dropout_rate = 0.1,
    config
  ) {
    super({
      name: "Transformer",
      inputs: tf.layers.dense({ units: input_vocab_size }),
      outputs: tf.layers
        .dense({ units: target_vocab_size })
        .apply(
          new Encoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            dropout_rate
          )
        )
        .apply(
          new Decoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            target_vocab_size,
            dropout_rate
          )
        ),
    });
    // this.encoder = new Encoder(
    //   num_layers,
    //   d_model,
    //   num_heads,
    //   dff,
    //   input_vocab_size,
    //   dropout_rate
    // );
    // this.decoder = new Decoder(
    //   num_layers,
    //   d_model,
    //   num_heads,
    //   dff,
    //   target_vocab_size,
    //   dropout_rate
    // );
    // this.final_layer = tf.layers.dense({ units: target_vocab_size });
  }

  call(inputs) {
    const [context, x] = inputs;

    const enc_output = this.encoder.apply(context); // (batch_size, context_len, d_model)
    const dec_output = this.decoder.apply([x, enc_output]); // (batch_size, target_len, d_model)

    // Final linear layer
    const logits = this.final_layer.apply(dec_output); // (batch_size, target_len, target_vocab_size)

    // Normally, TensorFlow.js doesn't require explicit handling of masks like in the Python version
    // so we don't do the "del logits._keras_mask" step

    return logits;
  }

  getClassName() {
    return "Transformer";
  }
}

export {
  positionalEncoding,
  PositionalEmbedding,
  MultiHeadAttention,
  CrossAttention,
  GlobalSelfAttention,
  CausalSelfAttention,
  Encoder,
  DecoderLayer,
  Decoder,
  Transformer,
};
