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
    this.posEncoding = positionalEncoding(2048, dModel); // Assume positionalEncoding is a function you've defined
  }

  computeMask(inputs, mask = null) {
    //   return this.embedding.computeMask(inputs, mask);
    return null;
  }

  call(inputs, kwargs) {
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
  
    scaledDotProductAttention(q, k, v, mask = null) {
      const matmulQK = tf.matMul(q, k.transpose([0, 1, 3, 2]));
      const dk = k.shape[k.shape.length - 1];
      let scaledAttentionLogits = matmulQK.div(tf.sqrt(dk));
  
      // Apply causal mask if required
      if (this.causal) {
        const seqLen = scaledAttentionLogits.shape[scaledAttentionLogits.shape.length - 2];
        const upperTriangular = tf.linalg.bandPart(tf.ones([seqLen, seqLen]), 0, -1);
        const identityMatrix = tf.eye(seqLen);
        const causalMask = upperTriangular.sub(identityMatrix);
        scaledAttentionLogits = scaledAttentionLogits.add(causalMask.mul(tf.scalar(-1e9)));
      }
  
      if (mask) {
        scaledAttentionLogits = scaledAttentionLogits.add(mask.mul(tf.scalar(-1e9)));
      }
  
      const attentionWeights = tf.softmax(scaledAttentionLogits, -1);
      const output = tf.matMul(attentionWeights, v);
  
      return [output, attentionWeights];
    }
  
    splitHeads(x, batch_size) {
      const reshaped = x.reshape([batch_size, -1, this.num_heads, this.depth]);
      return reshaped.transpose([0, 2, 1, 3]);
    }
  
    call(v, k, q, mask = null) {
      const batchSize = q.shape[0];
      const qProcessed = this.wq.apply(q);
      const kProcessed = this.wk.apply(k);
      const vProcessed = this.wv.apply(v);
  
      const qSplit = this.splitHeads(qProcessed, batchSize);
      const kSplit = this.splitHeads(kProcessed, batchSize);
      const vSplit = this.splitHeads(vProcessed, batchSize);
  
      const [scaledAttention, attentionWeights] = this.scaledDotProductAttention(qSplit, kSplit, vSplit, mask);
  
      const scaledAttentionTransposed = scaledAttention.transpose([0, 2, 1, 3]);
      const concatAttention = scaledAttentionTransposed.reshape([batchSize, -1, this.d_model]);
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
    const attnOutput = this.mha.apply(x, x, x);

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
  
    call(x, mask = null) {
      // Multi-head attention layer with causal mask enabled
      const [attnOutput, _] = this.mha.call(x, x, x, mask);
  
      // Add & normalize layer
      const addOutput = this.add.apply([x, attnOutput]);
      const output = this.layernorm.apply(addOutput);
  
      return output;
    }
  }

  

export {
  positionalEncoding,
  PositionalEmbedding,
  MultiHeadAttention,
  CrossAttention,
  GlobalSelfAttention,
  CausalSelfAttention,
};
